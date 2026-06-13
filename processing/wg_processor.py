"""
wg_processor.py
===============
Config-driven Wave Glider NRT NetCDF processor.

Reads all data sources defined in metadata_config.json, resamples onto a
common 10-min grid, merges into a single xr.Dataset, and writes a CF-compliant
NetCDF. ADCP is deliberately excluded (2D, separate file).

Usage
-----
    python wg_processor.py -i <data_dir> -o <output.nc> -c <config.json>

Example
-------
    python wg_processor.py \\
        -i /data/WHIRLS/NRT \\
        -o /data/WHIRLS/NRT/wg1169_WHIRLS_Mission3_L1.nc \\
        -c metadata_config.json
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

KNOTS_TO_MS = 0.514444


# ══════════════════════════════════════════════════════════════════════════════
# CADENCE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_cadence(df: pd.DataFrame,
                   max_gap_hard: pd.Timedelta,
                   min_dt: pd.Timedelta = pd.Timedelta("2min")) -> pd.Timedelta:
    """
    Estimate native reporting cadence from the data itself.
    Ignores debug bursts (< min_dt) and real outage gaps (>= max_gap_hard).
    Uses median of the remaining diffs — robust to mid-mission cadence changes.
    """
    diffs = df.index.to_series().diff().dropna()
    normal = diffs[(diffs > min_dt) & (diffs < max_gap_hard)]
    return normal.median() if not normal.empty else diffs.median()


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse timestamp column, set as sorted index, drop duplicates."""
    df["time"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = (df.dropna(subset=["time"])
            .drop(columns=[ts_col], errors="ignore")
            .set_index("time")
            .sort_index())
    return df[~df.index.duplicated(keep="last")]


def load_json_source(base_dir: Path, src_cfg: dict) -> Optional[pd.DataFrame]:
    """Load all JSON files matching glob pattern into a single DataFrame."""
    files = sorted(base_dir.glob(src_cfg["glob"]))
    if not files:
        return None

    records = []
    for f in files:
        try:
            raw = json.loads(f.read_text())
            key = src_cfg.get("json_key")
            if key:
                chunk = raw.get(key, [])
            elif isinstance(raw, list):
                chunk = raw
            else:
                # dict without explicit key — try recordData fallback
                chunk = raw.get("recordData", [])
            records.extend(chunk)
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    if not records:
        return None

    df = pd.DataFrame(records)

    # Parse timestamp first, before any numeric coercion
    ts_col = src_cfg["timestamp_col"]
    df = _parse_timestamp(df, ts_col)

    # Now coerce remaining columns to numeric if requested
    if src_cfg.get("to_float"):
        df = df.apply(pd.to_numeric, errors="coerce")

    return df


def load_csv_source(base_dir: Path, src_cfg: dict) -> Optional[pd.DataFrame]:
    """Load a single CSV file."""
    path = base_dir / src_cfg["filename"]
    if not path.exists():
        print(f"  [WARN] CSV not found: {path}")
        return None

    df = pd.read_csv(path)
    return _parse_timestamp(df, src_cfg["timestamp_col"])

def load_source(base_dir: Path, src_cfg: dict) -> Optional[pd.DataFrame]:
    """Dispatch to JSON or CSV loader based on source type."""
    if src_cfg["type"] == "json":
        return load_json_source(base_dir, src_cfg)
    elif src_cfg["type"] == "csv":
        return load_csv_source(base_dir, src_cfg)
    else:
        raise ValueError(f"Unknown source type: {src_cfg['type']}")


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN SELECTION & UNIT CONVERSIONS
# ══════════════════════════════════════════════════════════════════════════════

def select_columns(df: pd.DataFrame, src_cfg: dict) -> pd.DataFrame:
    """
    Keep only the columns we want.
    Priority: keep_vars list > keep_pattern regex > keep everything.
    Always drops drop_vars.
    """
    # Explicit keep list
    if "keep_vars" in src_cfg:
        keep = [c for c in src_cfg["keep_vars"] if c in df.columns]
        df = df[keep]

    # Regex pattern (e.g. amps_solar panelPower[1-4])
    elif "keep_pattern" in src_cfg:
        pat = re.compile(src_cfg["keep_pattern"])
        keep = [c for c in df.columns if pat.search(c)]
        df = df[keep]

    # Drop unwanted columns
    drop = src_cfg.get("drop_vars", [])
    return df.drop(columns=[c for c in drop if c in df.columns])


def apply_unit_conversions(df: pd.DataFrame, src_cfg: dict) -> pd.DataFrame:
    """Apply factor-based unit conversions and rename to target column name."""
    for src_col, conv in src_cfg.get("unit_conversions", {}).items():
        if src_col in df.columns:
            df[conv["target"]] = df[src_col] * conv["factor"]
            df = df.drop(columns=[src_col])
    return df


def apply_renames(df: pd.DataFrame, src_cfg: dict) -> pd.DataFrame:
    """Rename columns according to the rename map."""
    rename = {k: v for k, v in src_cfg.get("rename", {}).items() if k in df.columns}
    return df.rename(columns=rename)


# ══════════════════════════════════════════════════════════════════════════════
# GPS EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def ddmm_to_dd(val: float) -> float:
    """Convert DDMM.mmm GPS format to decimal degrees."""
    if np.isnan(val):
        return np.nan
    deg = int(abs(val) / 100)
    minutes = abs(val) - deg * 100
    dd = deg + minutes / 60
    return dd if val >= 0 else -dd


def extract_gps(df: pd.DataFrame, src_cfg: dict) -> Optional[xr.Dataset]:
    """
    Extract lat/lon from a source as a named GPS dataset.
    Returns None if no GPS columns configured.
    """
    lat_col = src_cfg.get("lat_col")
    lon_col = src_cfg.get("lon_col")
    gps_name = src_cfg.get("gps_name")

    if not (lat_col and lon_col and gps_name):
        return None
    if lat_col not in df.columns or lon_col not in df.columns:
        return None

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    # VeGAS stores GPS in DDMM.mmm format — convert if values look like it
    if lat.abs().max() > 90:
        lat = lat.apply(ddmm_to_dd)
        lon = lon.apply(ddmm_to_dd)

    return xr.Dataset({
        f"lat_{gps_name}": xr.DataArray(
            lat.values, dims="time", coords={"time": df.index}
        ),
        f"lon_{gps_name}": xr.DataArray(
            lon.values, dims="time", coords={"time": df.index}
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLATION ONTO MASTER GRID
# ══════════════════════════════════════════════════════════════════════════════

def regrid(df: pd.DataFrame,
           master_grid: pd.DatetimeIndex,
           src_cfg: dict,
           min_dt: pd.Timedelta) -> xr.Dataset:
    """
    Reindex DataFrame onto master_grid and interpolate/fill gaps.

    - Linear interpolation for continuous variables (max_gap auto-detected).
    - Forward-fill for discrete/flag variables (ffill_vars list).
    - Variables named in ffill_vars always use ffill regardless of method.
    """
    method       = src_cfg.get("interp_method", "linear")
    max_gap_hard = pd.Timedelta(src_cfg.get("interp_max_gap", "2h"))
    gap_tol      = src_cfg.get("gap_tolerance", 3)
    ffill_vars   = set(src_cfg.get("ffill_vars", []))

    # Detect native cadence (robust to outages and debug bursts)
    cadence  = detect_cadence(df, max_gap_hard, min_dt)
    max_gap  = min(cadence * gap_tol, max_gap_hard)

    # Cast bool columns to int8 (0/1) — bool + NaN = object after reindex, which breaks interpolation
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype("int8")

    # Defragment before xarray conversion (avoids PerformanceWarning on wide DataFrames)
    df = df.copy()

    # Split numeric vs non-numeric columns
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    grid_freq = master_grid.freq

    # Sources denser than or equal to the grid: resample to grid first
    # Sources sparser than the grid: reindex directly (gaps filled by interpolation below)
    cadence = detect_cadence(df, max_gap_hard, min_dt)

    if len(numeric_cols) > 0:
        df_num = df[numeric_cols]
        if cadence <= grid_freq.delta if hasattr(grid_freq, 'delta') else True:
            # Resample: bin observations into grid cells
            df_num = df_num.resample(grid_freq).mean()
        df_num = df_num.reindex(master_grid)
        ds = xr.Dataset.from_dataframe(df_num)
    else:
        ds = xr.Dataset(coords={"time": master_grid})

    for var in ds.data_vars:
        if var in ffill_vars:
            ds[var] = ds[var].ffill(dim="time")
        elif method == "ffill":
            ds[var] = ds[var].ffill(dim="time")
            # Zero out fills beyond max_gap
            gaps = pd.Series(master_grid).diff().fillna(pd.Timedelta(0))
            mask = xr.DataArray(gaps.values > max_gap, dims="time",
                                coords={"time": master_grid})
            ds[var] = ds[var].where(~mask)
        else:
            # Linear interpolation with max_gap cap
            ds[var] = ds[var].interpolate_na(
                dim="time",
                method="linear",
                max_gap=max_gap
            )

    # Add non-numeric columns (alarm flags, strings) via ffill only
    if non_numeric_cols:
        df_nn = df[non_numeric_cols]
        if cadence <= grid_freq.delta if hasattr(grid_freq, 'delta') else True:
            df_nn = df_nn.resample(grid_freq).last()
        df_nn = df_nn.reindex(master_grid)
        for col in non_numeric_cols:
            ds[col] = xr.DataArray(df_nn[col].ffill().values,
                                   dims="time", coords={"time": master_grid})

    return ds


# ══════════════════════════════════════════════════════════════════════════════
# CF METADATA
# ══════════════════════════════════════════════════════════════════════════════

def apply_cf_metadata(ds: xr.Dataset, cf_meta: dict) -> xr.Dataset:
    """Apply CF standard_name, units, long_name from config to matching variables."""
    for var in ds.data_vars:
        # Match exact name or strip _MIN/_MAX/_STDDEV suffix to find parent entry
        base = re.sub(r'_(MIN|MAX|STDDEV)$', '', var)
        attrs = cf_meta.get(var) or cf_meta.get(base)
        if attrs:
            # For non-MEAN stats, keep units from parent but update long_name
            if var != base:
                stat = var.split("_")[-1].lower()
                ds[var].attrs = {
                    "units":    attrs.get("units", ""),
                    "long_name": attrs.get("long_name", "").replace(
                        "mean", stat).replace("10-min mean", f"10-min {stat}")
                }
            else:
                ds[var].attrs = {k: v for k, v in attrs.items() if v}
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# GPS MERGING
# ══════════════════════════════════════════════════════════════════════════════

def build_merged_gps(gps_datasets: dict,
                     priority: list,
                     master_grid: pd.DatetimeIndex) -> xr.Dataset:
    """
    Merge GPS sources in priority order onto master grid.
    Stores each source as lat_<name>/lon_<name> data vars.
    Builds merged latitude/longitude coords (first non-NaN wins).
    """
    ds_gps = xr.Dataset()

    lat_merged = xr.DataArray(np.full(len(master_grid), np.nan),
                              dims="time", coords={"time": master_grid})
    lon_merged = xr.DataArray(np.full(len(master_grid), np.nan),
                              dims="time", coords={"time": master_grid})

    for name in priority:
        if name not in gps_datasets:
            continue
        ds_src = gps_datasets[name]
        lat_var = f"lat_{name}"
        lon_var = f"lon_{name}"

        # Interpolate this GPS source onto master grid
        lat = ds_src[lat_var].interp(time=master_grid, method="linear")
        lon = ds_src[lon_var].interp(time=master_grid, method="linear")

        # Store as data vars
        ds_gps[lat_var] = lat.assign_attrs({
            "units": "degrees_north", "long_name": f"Latitude from {name}",
            "standard_name": "latitude"
        })
        ds_gps[lon_var] = lon.assign_attrs({
            "units": "degrees_east", "long_name": f"Longitude from {name}",
            "standard_name": "longitude"
        })

        # Fill merged coords where still NaN
        lat_merged = lat_merged.fillna(lat)
        lon_merged = lon_merged.fillna(lon)

    ds_gps["latitude"]  = lat_merged.assign_attrs({
        "standard_name": "latitude", "units": "degrees_north", "axis": "Y",
        "long_name": "Latitude (merged: " + " > ".join(priority) + ")"
    })
    ds_gps["longitude"] = lon_merged.assign_attrs({
        "standard_name": "longitude", "units": "degrees_east", "axis": "X",
        "long_name": "Longitude (merged: " + " > ".join(priority) + ")"
    })

    return ds_gps


# ══════════════════════════════════════════════════════════════════════════════
# MASTER BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_master_dataset(input_dir: str, cfg: dict) -> Optional[xr.Dataset]:
    """
    Main orchestration. For each source in cfg['sources']:
      1. Load raw data
      2. Select / convert / rename columns
      3. Extract GPS if available
      4. Regrid onto master 10-min grid
      5. Merge into master dataset
    Then merge GPS, apply CF metadata, set global attributes.
    """
    base_dir    = Path(input_dir)
    mission_cfg = cfg["mission_config"]
    min_dt      = pd.Timedelta(mission_cfg.get("min_dt", "2min"))
    grid_freq   = mission_cfg.get("grid_freq", "10min")

    # ── Master time grid: CR6 is the anchor ──────────────────────────────────
    # We build it after loading CR6; fall back to telemetry if CR6 absent.
    master_grid = None
    datasets    = {}
    gps_sources = {}

    print(f"\n{'='*60}")
    print(f"  Building {mission_cfg['mission_name']} — {mission_cfg['platform_name']}")
    print(f"  Grid: {grid_freq}  |  min_dt filter: {min_dt}")
    print(f"{'='*60}\n")

    for src_name, src_cfg in cfg["sources"].items():
        print(f"► {src_name}")

        # 1. Load
        df = load_source(base_dir, src_cfg)
        if df is None or df.empty:
            print(f"  [SKIP] No data found\n")
            continue
        print(f"  Loaded {len(df)} records  "
              f"({df.index[0].strftime('%Y-%m-%d %H:%M')} → "
              f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")

        # 2. Select columns
        df = select_columns(df, src_cfg)

        # 3. Extract GPS before dropping lat/lon
        gps_ds = extract_gps(df, src_cfg)
        if gps_ds is not None:
            gps_sources[src_cfg["gps_name"]] = gps_ds
            print(f"  GPS stored as: lat_{src_cfg['gps_name']} / lon_{src_cfg['gps_name']}")

        # Drop lat/lon from main data (they go into GPS store only)
        for col in [src_cfg.get("lat_col"), src_cfg.get("lon_col")]:
            if col and col in df.columns:
                df = df.drop(columns=[col])

        # 4. Unit conversions + rename
        df = apply_unit_conversions(df, src_cfg)
        df = apply_renames(df, src_cfg)

        # 5. Build master grid from telemetry first, fall back to CR6
        if master_grid is None and src_name == "telemetry":
            start = df.index.min().floor(grid_freq)
            end   = df.index.max().ceil(grid_freq)
            master_grid = pd.date_range(start=start, end=end, freq=grid_freq, name="time")
            print(f"  Master grid: {len(master_grid)} steps at {grid_freq}")

        # ── Source-specific post-processing ──────────────────────────────────
        if src_name == "vegas":
            # Atmosphere pCO2: -50.0 is the calibration cycle sentinel
            if "VEGAS_PCO2_ATM" in df.columns:
                df["VEGAS_PCO2_ATM"] = df["VEGAS_PCO2_ATM"].where(df["VEGAS_PCO2_ATM"] > 0)
            # Ocean pCO2: >700 µatm is pre-equilibration startup spike
            if "VEGAS_PCO2_OCN" in df.columns:
                df["VEGAS_PCO2_OCN"] = df["VEGAS_PCO2_OCN"].where(df["VEGAS_PCO2_OCN"] < 700)

        datasets[src_name] = df
        print()

    # Fallback: build grid from telemetry if CR6 absent
    if master_grid is None:
        for fallback in ["telemetry", "airsea_cr6", "weather_airmar"]:
            if fallback in datasets:
                df_fb = datasets[fallback]
                master_grid = pd.date_range(
                    start=df_fb.index.min().floor(grid_freq),
                    end=df_fb.index.max().ceil(grid_freq),
                    freq=grid_freq,
                    name="time"
                )
                print(f"[INFO] Master grid built from {fallback}: "
                      f"{len(master_grid)} steps")
                break

    if master_grid is None:
        print("❌ Cannot build master grid — no data sources loaded.")
        return None

    # ── Regrid all sources onto master grid ──────────────────────────────────
    print("► Regridding all sources onto master grid...")
    ds_list = []
    for src_name, df in datasets.items():
        src_cfg = cfg["sources"][src_name]
        try:
            ds = regrid(df, master_grid, src_cfg, min_dt)
            ds_list.append(ds)
            print(f"  {src_name}: {len(ds.data_vars)} variables")
        except Exception as e:
            print(f"  [WARN] {src_name} regrid failed: {e}")

    if not ds_list:
        print("❌ No datasets to merge.")
        return None

    # ── Merge ────────────────────────────────────────────────────────────────
    print("\n► Merging...")
    ds_master = xr.merge(ds_list, join="outer", compat="override")

    # ── GPS ──────────────────────────────────────────────────────────────────
    print("► Building GPS coordinates...")
    priority = cfg.get("gps_priority", list(gps_sources.keys()))
    ds_gps   = build_merged_gps(gps_sources, priority, master_grid)
    ds_master = xr.merge([ds_master, ds_gps], join="outer", compat="override")
    ds_master = ds_master.set_coords(["latitude", "longitude"])

    # ── Trim to last valid telemetry point ───────────────────────────────────
    # Telemetry sets the master grid but may have trailing NaNs beyond last
    # real record — clip the dataset there rather than keeping empty time steps.
    tel_var = next((v for v in ds_master.data_vars if v.endswith("_TEL")
                    and ds_master[v].notnull().any()), None)
    if tel_var is not None:
        last_valid = pd.Timestamp(
            ds_master.time.values[ds_master[tel_var].notnull().values][-1]
        )
        ds_master = ds_master.sel(time=slice(None, last_valid))
        print(f"► Trimmed to last valid telemetry: {last_valid}")

    # ── Time slice ───────────────────────────────────────────────────────────
    start = mission_cfg.get("start_time")
    end   = mission_cfg.get("end_time")
    if start or end:
        ds_master = ds_master.sel(time=slice(start, end))

    # ── Drop all-NaN variables ───────────────────────────────────────────────
    empty = [v for v in ds_master.data_vars if ds_master[v].isnull().all()]
    if empty:
        print(f"\n► Dropping empty variables ({len(empty)}): {empty}")
        ds_master = ds_master.drop_vars(empty)

    # ── CF metadata ──────────────────────────────────────────────────────────
    print("► Applying CF metadata...")
    ds_master = apply_cf_metadata(ds_master, cfg.get("cf_metadata", {}))

    ds_master["time"].attrs = {
        "standard_name": "time",
        "axis":          "T",
        "long_name":     f"Time (UTC, {grid_freq} grid)"
    }

    # ── Global attributes ────────────────────────────────────────────────────
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    ds_master.attrs = {
        **cfg.get("global_attributes", {}),
        "title":               f"{mission_cfg['mission_name']} — {mission_cfg['platform_name']}",
        "id":                  f"WG_{mission_cfg['platform_id']}_{mission_cfg['mission_name'].replace(' ','_')}_L1",
        "history":             f"Created {now} by wg_processor.py",
        "time_coverage_start": str(ds_master.time.values[0])[:19],
        "time_coverage_end":   str(ds_master.time.values[-1])[:19],
        "time_coverage_resolution": grid_freq,
        "geospatial_lat_min":  float(ds_master["latitude"].min()),
        "geospatial_lat_max":  float(ds_master["latitude"].max()),
        "geospatial_lon_min":  float(ds_master["longitude"].min()),
        "geospatial_lon_max":  float(ds_master["longitude"].max()),
    }

    n_vars = len(ds_master.data_vars)
    n_times = len(ds_master.time)
    print(f"\n✅ Done — {n_vars} variables × {n_times} time steps "
          f"({ds_master.time.values[0]} → {ds_master.time.values[-1]})")

    return ds_master


def plot_data_coverage(ds: xr.Dataset, output_path: Optional[str] = None) -> None:
    """
    Horizontal bar chart showing % data coverage per variable.
    Bars are colored by sensor group inferred from variable name prefix/suffix.
    Sorted by coverage descending within each group.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    vars_1d = [v for v in ds.data_vars if ds[v].dims == ("time",)]
    if not vars_1d:
        print("[WARN] No 1D variables to plot coverage for.")
        return

    # Coverage fraction per variable
    n = len(ds.time)
    coverage = {v: float((~ds[v].isnull()).sum()) / n * 100 for v in vars_1d}

    # Sensor group colour map — keyed on variable name substrings
    GROUP_COLORS = {
        "GILL":   "#378ADD",
        "WXT":    "#BA7517",
        "LEGATO": "#1D9E75",
        "CODA":   "#5DCAA5",
        "CYCLOPS":"#2A9D2A",
        "CGR4":   "#E76F51",
        "SPN1":   "#F4A261",
        "WAVE":   "#7F77DD",
        "AIRMAR": "#EF9F27",
        "TEL":    "#888780",
        "AMPS":   "#E24B4A",
        "VEGAS":  "#8B4513",
        "ALARM":  "#F09595",
    }
    DEFAULT_COLOR = "#B4B2A9"

    def get_color(vname):
        for key, col in GROUP_COLORS.items():
            if key in vname:
                return col
        return DEFAULT_COLOR

    # Sort by coverage descending
    sorted_vars = sorted(coverage, key=lambda v: coverage[v], reverse=True)
    values      = [coverage[v] for v in sorted_vars]
    colors      = [get_color(v) for v in sorted_vars]

    n_vars = len(sorted_vars)
    bar_h  = 0.55
    fig_h  = max(6, n_vars * (bar_h + 0.15))

    fig, ax = plt.subplots(figsize=(11, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_edgecolor("#D3D1C7")

    bars = ax.barh(range(n_vars), values, height=bar_h, color=colors, alpha=0.85)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 2:
            ax.text(val + 0.5, i, f"{val:.0f}%",
                    va="center", ha="left", fontsize=7, color="#5F5E5A")

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(sorted_vars, fontsize=7)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Data coverage (%)", fontsize=9, color="#5F5E5A")
    ax.tick_params(colors="#888780", labelsize=7)
    ax.axvline(100, color="#D3D1C7", lw=0.8, ls=":")

    # Legend
    seen = {}
    for v in sorted_vars:
        for key, col in GROUP_COLORS.items():
            if key in v and key not in seen:
                seen[key] = col
    patches = [mpatches.Patch(color=c, alpha=0.85, label=k) for k, c in seen.items()]
    ax.legend(handles=patches, fontsize=7, frameon=False,
              loc="lower right", ncol=2)

    t0 = str(ds.time.values[0])[:10]
    t1 = str(ds.time.values[-1])[:10]
    ax.set_title(f"Data coverage — {t0} to {t1}  ({n} time steps × {n_vars} variables)",
                 fontsize=10, color="#5F5E5A", pad=8)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"📊 Coverage plot → {output_path}")
    else:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Glider CF-NetCDF Processor")
    parser.add_argument("-i", "--input",  required=True, help="Input data directory")
    parser.add_argument("-o", "--output", required=True, help="Output NetCDF path")
    parser.add_argument("-c", "--config", required=True, help="Path to metadata_config.json")
    args = parser.parse_args()

    try:
        cfg = json.loads(Path(args.config).read_text())
    except Exception as e:
        print(f"❌ Config error: {e}")
        sys.exit(1)

    ds = build_master_dataset(args.input, cfg)

    if ds is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out, format="NETCDF4")
        print(f"💾 Saved → {out}")
        plot_data_coverage(ds, output_path=str(out).replace(".nc", "_coverage.png"))
    else:
        sys.exit(1)