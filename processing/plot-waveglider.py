"""
plot-waveglider.py
==================
NRT multi-panel figure for Wave Glider WHIRLS Mission 3.
Reads entirely from the L1 NetCDF produced by wg_processor.py.

WHAT TO CHANGE
--------------
  STYLE          → "classic" | "clean"
  NC / OUT       → paths
  BAT_WARN/CRIT  → battery thresholds (Wh)
  build_panels() → all panel definitions: variables, colours, y-limits
  NTICKS         → number of y-ticks on every axis (keeps twinx grids aligned)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← change things here
# ══════════════════════════════════════════════════════════════════════════════

STYLE = "clean"   # "classic" | "clean"

NC  = Path("/Users/xedhjo/Documents/Data/WHIRLS/WHIRLS Mission 3 - NRT/wg1169_WHIRLS_Mission3_L1.nc")
OUT = Path("/Users/xedhjo/Documents/Data/WHIRLS/WHIRLS Mission 3 - NRT/waveglider_mission_3_nrt.png")

BAT_WARN = 250    # Wh — amber threshold
BAT_CRIT = 200    # Wh — red threshold

NTICKS   = 5      # y-tick count on every axis — keeps twinx grid lines aligned

# ══════════════════════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════════════════════

STYLES = {
    "classic": {
        "rc":          {"font.size": 10, "axes.spines.top": False, "axes.linewidth": 0.8},
        "grid_alpha":  0.4,  "grid_ls": ":",  "grid_color": "0.7",
        "title_color": "black", "label_color": "black", "tick_color": "black",
        "bg":          "white", "spine_color": "black",
        "C": {
            "green_mid": "#1D9E75", "blue_mid": "#378ADD", "amber": "#BA7517",
            "red": "#E24B4A", "teal": "#5DCAA5", "border": "#CCCCCC",
            "surface": "white", "muted": "0.4", "hint": "0.6",
        },
    },
    "clean": {
        "rc": {
            "font.size": 11, "axes.spines.right": False, "axes.spines.top": False,
            "axes.linewidth": 1,
            "xtick.major.size": 8, "xtick.minor.size": 4,
            "ytick.major.size": 8, "ytick.minor.size": 4,
            "xtick.major.width": 1, "xtick.minor.width": 1,
            "ytick.major.width": 1, "ytick.minor.width": 1,
        },
        "grid_alpha":  1.0,  "grid_ls": "-",  "grid_color": "#E0DDD6",
        "title_color": "#000000",
        "label_color": "#000000",
        "tick_color":  "#000000",
        "bg":          "#F7F5F0", "spine_color": "#000000",
        "C": {
            "green_mid": "#1D9E75", "blue_mid": "#378ADD", "amber": "#BA7517",
            "red": "#E24B4A", "teal": "#5DCAA5", "border": "#E0DDD6",
            "surface": "#FFFFFF", "muted": "#444442", "hint": "#888780",
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  PANEL DEFINITIONS  ← edit ylims, colours, variables here
# ══════════════════════════════════════════════════════════════════════════════
# Each variable entry:
#   ax_idx  : 0 = left axis, 1 = right axis, 2 = far-right axis (outward 60pt)
#   var     : xarray DataArray
#   color   : hex or named
#   ls      : "-" | "--" | ":" | "fill" (area fill) | "scatter" (dots)
#   ylim    : (ymin, ymax)   — sets axis limits; NTICKS evenly spaced ticks applied
#   label   : legend label
#   ylabel  : axis label (only first var on each ax_idx sets the ylabel)

def build_panels(ds):
    # Derived fields
    gill_horiz = np.sqrt(ds["WIND_U_GILL_MEAN"]**2 + ds["WIND_V_GILL_MEAN"]**2)
    has_pco2   = "VEGAS_PCO2_OCN" in ds

    return [
        {
            "title": "Radiation",
            "vars": {
                "cgr": (0, ds["CGR4_RAD_MEAN"],     "#E76F51", "-",    (0, 1000), "CGR4 LW ↓",     "LW Radiation (W/m²)"),
                "spt": (1, ds["SPN1_TOTAL_MEAN"],    "#F4A261", "-",    (0, 1000), "SPN1 SW total",  "SW Radiation (W/m²)"),
                "spd": (1, ds["SPN1_DIFFUSE_MEAN"],  "#F4A261", ":",    (0, 1000), "SPN1 SW diffuse","SW Radiation (W/m²)"),
            }
        },
        {
            "title": "Winds & Waves",
            "vars": {
                "vw":  (0, ds["WIND_SPEED_WXT_MEAN"],      "k",       "-",    (0, 20),   "Vaisala wind",  "Wind Speed (m/s)"),
                "aw":  (0, ds["WIND_SPEED_AIRMAR_MEAN"],   "k",       ":",    (0, 20),   "Airmar wind",   "Wind Speed (m/s)"),
                "gh":  (0, gill_horiz,                      "k",       "--",   (0, 20),   "Gill horiz",    "Wind Speed (m/s)"),
                "gz":  (1, ds["WIND_W_GILL_MEAN"],          "0.5",     "-",    (-2, 2),   "Gill Uz",       "Vertical Wind (m/s)"),
                "wv":  (2, ds["WAVE_SIGNIFICANT_HEIGHT"],   "#0A9396", "fill", (0, 6),    "Wave height",   "Wave Height (m)"),
            }
        },
        {
            "title": "Surface Met",
            "vars": {
                "vp":  (0, ds["BARO_PRES_WXT_MEAN"],        "0.4",     "-",    (940, 1040), "Pressure (WXT)",  "Pressure (hPa)"),
                "ap":  (0, ds["BARO_PRES_AIRMAR_MEAN"],     "0.4",     ":",    (940, 1040), "Pressure (Airmar)","Pressure (hPa)"),
                "rh":  (1, ds["RH_WXT_MEAN"],               "#C2B280", "-",    (50, 100),   "RH",              "RH (%)"),
                "rn":  (2, ds["RAIN_INTENSITY_WXT_MEAN"],   "#005F73", "fill", (0, 5),      "Rain",            "Rain (mm/hr)"),
            }
        },
        {
            "title": "Air-Sea Interface",
            "vars": {
                "ta":  (0, ds["TEMP_AIR_WXT_MEAN"],         "#E9724C", "-",    (5, 25),    "T air (WXT)",    "Temperature (°C)"),
                "aa":  (0, ds["TEMP_AIR_AIRMAR_MEAN"],      "#E9724C", ":",    (5, 25),    "T air (Airmar)", "Temperature (°C)"),
                "tw":  (0, ds["TEMP_WATER_LEGATO_MEAN"],    "#BA0B2F", "--",   (5, 25),    "SST (RBR)",      "Temperature (°C)"),
                "ss":  (1, ds["SAL_LEGATO_MEAN"],           "#4895EF", "-",    (34, 37),   "Salinity",       "Salinity (psu)"),
            }
        },
        {
            "title": "Biogeochemistry & pCO₂",
            "vars": {
                "ox":  (0, ds["O2_CONC_CODA_MEAN"],         "#003049", "-",    (150, 400), "O₂",           "Oxygen (µmol/L)"),
                "ch":  (1, ds["CHLOR_CYCLOPS_MEAN"],        "#2A9D2A", "fill", (0, 10),    "Chlorophyll",  "Chl (mg/m³)"),
                **({
                    "po": (2, ds["VEGAS_PCO2_OCN"],         "#8B4513", "-",    (300, 700), "pCO₂ ocean",   "pCO₂ (µatm)"),
                    "pa": (2, ds["VEGAS_PCO2_ATM"],          "#8B4513", "scatter",(300, 700),"pCO₂ atm",    "pCO₂ (µatm)"),
                } if has_pco2 else {}),
            }
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def style_ax(ax, S):
    ax.set_facecolor(S["C"]["surface"])
    for sp in ax.spines.values():
        sp.set_edgecolor(S["spine_color"])
    ax.tick_params(colors=S["tick_color"], which="both")
    ax.grid(axis="y", color=S["grid_color"], lw=0.5,
            ls=S["grid_ls"], alpha=S["grid_alpha"], zorder=0)
    ax.yaxis.label.set_color(S["label_color"])
    ax.title.set_color(S["title_color"])


def set_ylim_nticks(ax, ylim, nticks=NTICKS):
    """Set ylim and place exactly nticks evenly-spaced ticks.
    Using the same nticks on every twinx axis keeps horizontal grid lines aligned."""
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.LinearLocator(nticks))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))


def shade_days(ax, days, C):
    for i, d in enumerate(days):
        if i % 2 == 0:
            ax.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta("1D"),
                       color=C["border"], alpha=0.15, zorder=0, lw=0)


def daily_energy(series, index):
    dt_h = index.to_series().diff().dt.total_seconds().fillna(3600) / 3600
    return (series * dt_h).groupby(index.date).sum()


def add_daily_stats(ax, sol_d, out_d, net_d, batt_wh, times_pd, S):
    """
    Three side-by-side boxes: previous day, current day, mission average.
    Days with <50% coverage get dashed border but still show values.
    Mission average skips the first day (thruster kill) and incomplete days.
    """
    C = S["C"]
    MIN_RECORDS = 72   # 50% of a full day at 10-min cadence

    records_per_day = times_pd.normalize().value_counts()

    def _box(x, txt, border_col, dashed=False):
        ls = (0, (4, 3)) if dashed else "solid"
        ax.text(x, 0.04, txt,
                transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=C["muted"], fontfamily="monospace", zorder=5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=C["surface"],
                          edgecolor=border_col, linewidth=1.5, linestyle=ls))

    def _day_txt(day):
        n       = records_per_day.get(pd.Timestamp(day), 0)
        partial = n < MIN_RECORDS
        net     = net_d.get(day, 0)
        sign    = "+" if net >= 0 else "-"
        mask    = times_pd.normalize() == pd.Timestamp(day)
        b       = batt_wh[mask]
        bmin, bmax = (float(b.min()), float(b.max())) if len(b) else (float("nan"),)*2
        border  = (C["green_mid"] if net >= 0 else C["amber"] if net > -25 else C["red"])
        txt = (f"Solar in:    {sol_d.get(day,0):>5.0f} Wh\n"
               f"Output:     -{abs(out_d.get(day,0)):>5.0f} Wh\n"
               f"Net charge: {sign}{abs(net):>5.0f} Wh\n"
               f"Range:     {bmin:.0f}–{bmax:.0f} Wh")
        return txt, border, partial

    # Previous and current day
    last_two = sorted(sol_d.index)[-2:]
    for day, label, x in zip(last_two, ["Previous day", "Current day"], [0.72, 0.87]):
        txt, border, partial = _day_txt(day)
        _box(x, f"{label}\n{txt}", border, dashed=partial)

    # Mission average — skip first day and incomplete days
    all_days   = sorted(sol_d.index)
    good = [d for d in all_days[1:]   # skip first day (thruster kill)
            if records_per_day.get(pd.Timestamp(d), 0) >= MIN_RECORDS]
    if len(good) >= 2:
        avg_sol = np.mean([sol_d.get(d, 0) for d in good])
        avg_out = np.mean([out_d.get(d, 0) for d in good])
        avg_net = avg_sol - avg_out
        sign    = "+" if avg_net >= 0 else "-"
        border  = (C["green_mid"] if avg_net >= 0 else C["amber"] if avg_net > -25 else C["red"])
        txt = (f"Mission avg  ({len(good)}d)\n"
               f"Solar in:    {avg_sol:>5.0f} Wh\n"
               f"Output:     -{abs(avg_out):>5.0f} Wh\n"
               f"Net charge: {sign}{abs(avg_net):>5.0f} Wh\n"
               f"")
        _box(0.57, txt, border)


def panel_disagreeing_days(ds, panel_ids):
    """Return {date: [faulty panel ids]} for panels outputting < 50% of mean."""
    DAY_THRESH, BAD_FRAC = 2.0, 0.30
    stack     = np.vstack([ds[f"SOLAR_W_PANEL{i}_AMPS"].fillna(0).values for i in panel_ids])
    times     = pd.to_datetime(ds.time.values)
    is_day    = stack.max(axis=0) > DAY_THRESH
    panel_mean = stack.mean(axis=0)
    result = {}
    for day in sorted(set(t.date() for t in times)):
        mask     = np.array([t.date() == day for t in times])
        day_mask = mask & is_day
        if day_mask.sum() == 0:
            continue
        faulty = [pid for pid, row in zip(panel_ids, stack)
                  if (row[day_mask] < panel_mean[day_mask] * 0.5).mean() > BAD_FRAC]
        if faulty:
            result[day] = faulty
    return result


def set_xaxis(ax, times):
    # days = sorted({pd.Timestamp(t).date() for t in times
    #                if TIME_START <= pd.Timestamp(t) <= TIME_END + pd.Timedelta("1D")})
    # ax.set_xticks([pd.Timestamp(d) for d in days])
    # ax.set_xticklabels([d.strftime("%-d %b") for d in days],
    #                    fontsize=11, fontweight="medium")
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(""))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

ds_raw = xr.open_dataset(NC)

# Regrid to regular 10-min, interpolate gaps up to 2h
t_reg = pd.date_range(
    start=pd.Timestamp(ds_raw.time.values[0]).floor("10min"),
    end=pd.Timestamp(ds_raw.time.values[-1]).ceil("10min"),
    freq="10min"
)
ds = ds_raw.reindex(time=t_reg).interpolate_na(dim="time", max_gap=np.timedelta64(120, "m"))

# Power signals (all from NC now)
solar_w   = ds["SOLAR_W_AMPS"].fillna(0)
output_w  = ds["OUTPUT_W_AMPS"].fillna(0)
charge_w  = ds["CHARGE_W_AMPS"].fillna(0)
batt_wh = ds[["BATT_WH_TEL", "BATT_WH_AMPS"]].to_array().max("variable").fillna(0)
times     = pd.to_datetime(ds.time.values)
days      = sorted({pd.Timestamp(t).date() for t in times})

# x-axis bounds from telemetry — most continuous source, covers full mission
# fall back to full time array if telemetry absent
if "HEADING_TEL" in ds and ds["HEADING_TEL"].notnull().any():
    tel_times  = pd.to_datetime(ds.time.values[ds["HEADING_TEL"].notnull().values])
    TIME_START = tel_times[0]
    TIME_END   = tel_times[-1]
else:
    TIME_START = times[0]
    TIME_END   = times[-1]

# Solar panel check
panel_ids = [i for i in range(1, 5)
             if f"SOLAR_W_PANEL{i}_AMPS" in ds
             and float(ds[f"SOLAR_W_PANEL{i}_AMPS"].max()) > 0.01]
sol_daily        = daily_energy(solar_w.to_series(), pd.DatetimeIndex(times))
sol_d            = daily_energy(solar_w.to_series(),  pd.DatetimeIndex(times))
out_d            = daily_energy(output_w.to_series(), pd.DatetimeIndex(times))
net_d            = sol_d - out_d
disagreeing_days = panel_disagreeing_days(ds, panel_ids) if panel_ids else {}


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_figure():
    S = STYLES[STYLE]
    plt.rcParams.update(S["rc"])

    panels  = build_panels(ds)
    n_total = len(panels) + 2   # airsea panels + power + battery

    fig, axs = plt.subplots(
        nrows=n_total, ncols=1,
        figsize=(14, 3.2 * len(panels) + 4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1] * len(panels) + [1, 1]}
    )
    fig.patch.set_facecolor(S["bg"])

    # ── Science panels ────────────────────────────────────────────────────────
    for panel_idx, panel in enumerate(panels):
        ax        = axs[panel_idx]
        twin_r    = ax.twinx()
        twin_rr   = ax.twinx()
        twin_rr.spines["right"].set_position(("outward", 60))
        all_axes  = [ax, twin_r, twin_rr]
        ylabels   = {}   # ax_idx -> ylabel (first one wins)
        handles   = []

        for name, (ax_idx, data, color, ls, ylim, label, ylabel) in panel["vars"].items():
            cur_ax = all_axes[ax_idx]

            # Plot
            if ls == "fill":
                cur_ax.fill_between(data.time, data, 0, fc=color, alpha=0.15, ec=None)
                cur_ax.plot(data.time, data, color=color, lw=1, alpha=0.3)
                handles.append(mpatches.Patch(color=color, alpha=0.5, label=label))
            elif ls == "scatter":
                cur_ax.scatter(data.time, data, color=color, s=8, alpha=0.7, linewidths=0)
                handles.append(mlines.Line2D([], [], color=color, marker=".", ms=6,
                                             lw=0, label=label))
            else:
                lw = 2.0 if ls == "-" else 1.2
                cur_ax.plot(data.time, data, color=color, lw=lw, ls=ls)
                handles.append(mlines.Line2D([], [], color=color, ls=ls, lw=lw, label=label))

            # Limits & ticks — same NTICKS on every axis keeps grid lines aligned
            set_ylim_nticks(cur_ax, ylim)
            if ax_idx not in ylabels:
                cur_ax.set_ylabel(ylabel)
                ylabels[ax_idx] = ylabel

            # Styling
            cur_ax.set_zorder(10 - 4 * ax_idx)
            cur_ax.patch.set_visible(False)
            cur_ax.spines["top"].set_visible(False)
            style_ax(cur_ax, S)
            if ax_idx in (1, 2):
                cur_ax.spines["right"].set_visible(True)
                cur_ax.spines["right"].set_edgecolor(S["spine_color"])

        ax.grid(axis="y", ls=S["grid_ls"], alpha=S["grid_alpha"], color=S["grid_color"])
        ax.set_title(panel["title"], loc="left", fontsize="large", color=S["title_color"])
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1),
                  ncol=5, frameon=False, fontsize="small")

        # Hide unused twin axes
        used_ax_idxs = {v[0] for v in panel["vars"].values()}
        for i, a in enumerate(all_axes):
            if i not in used_ax_idxs:
                a.axis("off")

    # ── Power panel ───────────────────────────────────────────────────────────
    ax_pwr = axs[len(panels)]
    style_ax(ax_pwr, S)

    charge_pos = np.where(charge_w >= 0, charge_w, 0)
    charge_neg = np.where(charge_w <  0, charge_w, 0)

    ax_pwr.fill_between(times, solar_w,    alpha=0.12, color=S["C"]["green_mid"], zorder=1)
    ax_pwr.fill_between(times, output_w,   alpha=0.10, color=S["C"]["blue_mid"],  zorder=1)
    ax_pwr.fill_between(times, charge_pos, alpha=0.15, color=S["C"]["green_mid"], zorder=1)
    ax_pwr.fill_between(times, charge_neg, alpha=0.15, color=S["C"]["red"],       zorder=1)
    ax_pwr.plot(times, solar_w,  color=S["C"]["green_mid"], lw=2.0, zorder=3)
    ax_pwr.plot(times, output_w, color=S["C"]["blue_mid"],  lw=2.0, zorder=3)
    ax_pwr.plot(times, charge_w, color=S["C"]["amber"],     lw=1.6, zorder=3,
                ls="--", dashes=(4, 3))
    ax_pwr.axhline(0, color="0.7", lw=0.9, zorder=2)

    set_ylim_nticks(ax_pwr, (-100, 100))
    ax_pwr.set_ylabel("Power  (W)")
    ax_pwr.set_title("Power", loc="left", fontsize="large", color=S["title_color"])
    ax_pwr.set_title(
        f"Battery  {float(batt_wh[-1]):.0f} / {float(batt_wh.max()):.0f} Wh"
        f"  ·  {pd.Timestamp(times[-1]).strftime('%-d %b %H:%M')} UTC",
        loc="right", fontsize=12, color=S["C"]["muted"], y=0.9)
    ax_pwr.legend(handles=[
        mpatches.Patch(facecolor=S["C"]["green_mid"], alpha=0.7, label="Solar in"),
        mpatches.Patch(facecolor=S["C"]["blue_mid"],  alpha=0.7, label="Output"),
        plt.Line2D([0],[0], color=S["C"]["amber"], lw=1.6, ls="--", dashes=(4,3),
                   label="Net charge (+) / draw (−)"),
    ], fontsize="small", frameon=False, loc="lower right",
       ncol=3, bbox_to_anchor=(1, 1))

    add_daily_stats(ax_pwr, sol_d, out_d, net_d, batt_wh, pd.DatetimeIndex(times), S)

    # ── Battery panel ─────────────────────────────────────────────────────────
    ax_bat = axs[len(panels) + 1]
    style_ax(ax_bat, S)

    batt_raw    = batt_wh.values                                          # raw — for annotation
    batt_smooth = (batt_wh.rolling(time=6, center=True, min_periods=1)   # ~1h smooth for plot
                         .mean().values)

    for i in range(len(times) - 1):
        col = (S["C"]["green_mid"] if batt_smooth[i] > BAT_WARN
               else S["C"]["amber"] if batt_smooth[i] > BAT_CRIT
               else S["C"]["red"])
        ax_bat.plot(times[i:i+2], batt_smooth[i:i+2], color=col, lw=2.2, zorder=3)

    ax_bat.fill_between(times, batt_smooth, alpha=0.12, color=S["C"]["teal"], zorder=1)
    ax_bat.axhline(BAT_WARN, color=S["C"]["amber"], lw=2, ls=":", zorder=2)
    ax_bat.axhline(BAT_CRIT, color=S["C"]["red"],   lw=2, ls=":", zorder=2)
    set_ylim_nticks(ax_bat, (0, 1000))
    ax_bat.set_ylabel("Battery  (Wh)")
    ax_bat.annotate(f"{batt_raw[-1]:.0f} Wh",
                    xy=(times[-1], batt_raw[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=12, color=S["C"]["muted"], va="center", fontweight="medium")

    if disagreeing_days:
        faulty    = sorted({p for ps in disagreeing_days.values() for p in ps})
        days_str  = ", ".join(d.strftime("%-d %b") for d in sorted(disagreeing_days))
        ax_bat.set_title(f"Panel(s) {faulty} low output: {days_str}",
                         loc="center", fontsize="large", color=S["C"]["red"])
    else:
        ax_bat.set_title("All solar panels working",
                         loc="center", fontsize="large", color=S["C"]["green_mid"])
    ax_bat.set_title("Battery charge", loc="left", fontsize="large", color=S["title_color"])

    # ── Shared x-axis & day shading ───────────────────────────────────────────
    for ax in axs:
        shade_days(ax, days, S["C"])

    set_xaxis(axs[-1], times)
    # axs[-1].set_xlabel("Time (UTC)", color=S["label_color"])
    axs[-1].set_xlim(
        pd.Timestamp(TIME_START.date()),
        pd.Timestamp(TIME_END.date()) + pd.Timedelta("1D")
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════════════

fig = build_figure()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, transparent=True, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"Saved → {OUT}")