"""fig_charts.py -- Figures 4-7 (data and method charts). Run: python fig_charts.py [4 5 6 7]"""
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figlib import C, HALO, REPO, TEXTWIDTH, audit, save, style

from data_sources import load_oil_prices, load_portwatch, load_reference
from event_study import placebo_cars, run_event_table
from flow_index import KN_TO_KM_PER_DAY, kalman_fuse, simulate_fusion_case, snapshot_to_flow
from s1_cfar import detect_vessels, score_detections, synth_scene

style()
CFG = yaml.safe_load(open(REPO / "config.yaml"))
ES = CFG["event_study"]
PX = load_oil_prices()
PW = load_portwatch("hormuz")
EV = load_reference("episode_onsets")
TAB = REPO / "outputs" / "tables"

EP_COL = {"EP6": C["ep4"], "EP5": C["ep3"], "EP4": C["ep2"], "EP3": "#B07A0F",
          "EP2": "#7C8790", "EP1": C["ep1"]}
EP_SHORT = dict(zip(EV.episode_id, EV.short_label))


def clean(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=C["rule"], lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def repel(ys, min_gap, lo=-np.inf, hi=np.inf):
    """Spread 1-D label positions so neighbours are >= min_gap apart."""
    order = np.argsort(ys)
    y = np.asarray(ys, float)[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < min_gap:
                push = (min_gap - (y[i] - y[i - 1])) / 2 + 1e-9
                y[i - 1] -= push; y[i] += push; moved = True
        y = np.clip(y, lo, hi)
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return out


# ============================================================== Figure 4
def fig4():
    """Two-panel timeline: Hormuz tanker transits and Brent, 2023 onward."""
    t0, t1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-11-08")
    war = pd.Timestamp("2026-02-28")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(TEXTWIDTH, 4.35), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.16))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.075)

    tank = PW["n_tanker"].loc[t0:]
    a1.fill_between(tank.index, 0, tank.values, color=C["navy"], alpha=0.10, lw=0, step="mid")
    a1.plot(tank.index, tank.values, color="#8FB0C8", lw=0.35)
    a1.plot(tank.index, tank.rolling(7, center=True).mean(), color=C["navy"], lw=1.2)
    a1.set_ylim(0, 95); a1.set_yticks([0, 20, 40, 60, 80])
    a1.set_ylabel("Tankers per day")

    b = PX["brent"].dropna().loc[t0:]
    a2.plot(b.index, b.values, color=C["orange"], lw=1.0)
    a2.set_ylim(50, 150); a2.set_yticks([60, 80, 100, 120, 140])
    a2.set_ylabel("Brent, USD per barrel")

    marks = [("2024-04-13", "Apr 2024"), ("2024-10-01", "Oct 2024"), ("2025-06-13", "Jun 2025"),
             ("2026-02-28", "28 Feb 2026")]
    for ax in (a1, a2):
        ax.axvspan(war, t1, color=C["ep4"], alpha=0.06, lw=0, zorder=0)
        for d, _ in marks:
            ax.axvline(pd.Timestamp(d), color=C["ink"], lw=0.55, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(t0, t1)
        clean(ax)
    for d, lab in marks:
        a1.text(pd.Timestamp(d) - pd.Timedelta(days=8), 92, lab, ha="right", va="top", fontsize=6.4,
                color=C["ink"])

    # annotations with real numbers
    pre = PW["n_tanker"].loc["2025-03-01":"2026-02-28"].mean()
    post = PW["n_tanker"].loc["2026-03-01":].mean()
    a1.plot([pd.Timestamp("2025-03-01"), pd.Timestamp("2026-02-27")], [pre, pre], color=C["navy"],
            lw=0.7, ls=(0, (1, 1.2)), zorder=3)
    a1.text(pd.Timestamp("2025-10-20"), 3, f"mean {pre:.1f} per day\nMar 2025 – Feb 2026",
            fontsize=6.3, ha="center", va="bottom", color=C["navy"], linespacing=1.15,
            path_effects=HALO)
    a1.text(pd.Timestamp("2026-06-25"), 30, f"{post:.1f} per day\nsince 1 Mar 2026", fontsize=6.3,
            ha="center", va="bottom", color=C["ep4"], fontweight="semibold")
    pk = b.loc["2026-03-01":].idxmax()
    a2.annotate(f"Peak {b.max():.2f} on {pk:%-d %b %Y}", xy=(pk, b.max()),
                xytext=(pd.Timestamp("2025-12-20"), 141), fontsize=6.3, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.45, color=C["muted"], shrinkB=2,
                                relpos=(1.0, 0.5)))
    lo_jun = b.loc["2026-06-01":"2026-07-15"]
    a2.annotate(f"Low of {lo_jun.min():.2f} on {lo_jun.idxmin():%-d %b}", xy=(lo_jun.idxmin(), lo_jun.min()),
                xytext=(pd.Timestamp("2025-12-20"), 54), fontsize=6.3, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.45, color=C["muted"], shrinkB=2,
                                relpos=(1.0, 0.5)))
    a1.set_title("A   Strait of Hormuz tanker transits (IMF PortWatch, AIS-derived; line: 7-day mean)",
                 fontsize=7.6, pad=4)
    a2.set_title("B   Brent spot price (EIA)", fontsize=7.6, pad=4)
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    audit(fig, "fig4")
    save(fig, "fig4_timeline")
    plt.close(fig)


# ============================================================== Figure 5
def fig5():
    est, win = tuple(ES["estimation_window"]), tuple(ES["event_window"])
    _, paths = run_event_table(PX["brent"], EV, est=est, win=win, placebo_start=ES["placebo_start"])
    ks = np.arange(0, win[1] + 1)
    band = np.array([np.quantile(placebo_cars(PX["brent"], est=est, window=(0, int(k)),
                                              start=ES["placebo_start"], exclude=list(EV["day0"])),
                                 [0.05, 0.95]) for k in ks])
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.55))
    fig.subplots_adjust(left=0.085, right=0.83, top=0.975, bottom=0.13)
    ax.fill_between(ks, band[:, 0], band[:, 1], color="#D4DADF", alpha=0.75, lw=0, zorder=1)
    ax.axhline(0, color=C["ink"], lw=0.55, zorder=2)
    ax.axvline(0, color=C["ink"], lw=0.55, ls=(0, (2, 2)), zorder=2)
    order = ["EP1", "EP2", "EP3", "EP4", "EP5", "EP6"]
    ends = {}
    for ep in order:
        p = paths[paths.episode_id == ep]
        main = ep == "EP6"
        ax.plot(p.tau, p.car, color=EP_COL[ep], lw=2.0 if main else 1.05,
                zorder=4 if main else 3, solid_joinstyle="round")
        ax.scatter(p.tau.iloc[-1], p.car.iloc[-1], s=14 if main else 8, color=EP_COL[ep], zorder=5,
                   edgecolor="white", linewidth=0.5)
        ends[ep] = float(p.car.iloc[-1])
    ax.set_xlim(win[0], win[1] + 0.4)
    ax.set_ylim(-22, 60)
    ax.set_xticks(np.arange(-5, 21, 5))
    ax.set_xlabel("Trading days relative to event day 0")
    ax.set_ylabel("Cumulative abnormal return (log points, %)")
    clean(ax)
    # right-margin labels with repulsion (data units -> label slots)
    eps = list(ends)
    ys = repel([ends[e] for e in eps], min_gap=4.6, lo=-21, hi=58)
    for e, y in zip(eps, ys):
        ax.annotate(f"{EP_SHORT[e]}  {ends[e]:+.1f}", xy=(win[1], ends[e]), xycoords="data",
                    xytext=(win[1] + 1.3, y), textcoords="data", va="center", ha="left",
                    fontsize=6.7, color=EP_COL[e], fontweight="semibold" if e == "EP6" else "normal",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=EP_COL[e], shrinkA=0, shrinkB=2.5))
    ax.text(10.5, band[10, 1] + 2.0, "90% of placebo days", fontsize=6.3, color=C["muted"],
            ha="center", va="bottom")
    audit(fig, "fig5")
    save(fig, "fig5_event_study")
    plt.close(fig)


# ============================================================== Figure 6
def fig6():
    """Episode dumbbell: day-0 jump against day-20 outcome, with transit change."""
    t = pd.read_csv(TAB / "event_study_brent.csv", index_col=0)
    fr = pd.read_csv(TAB / "flow_response.csv", index_col=0)
    order = ["EP6", "EP5", "EP4", "EP3", "EP2", "EP1"]
    fig = plt.figure(figsize=(TEXTWIDTH, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=0.06,
                          left=0.205, right=0.985, top=0.86, bottom=0.17)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    y = np.arange(len(order))[::-1]
    for yi, ep in zip(y, order):
        c0, c20, p20 = t.loc[ep, "car_0_0"], t.loc[ep, "car_0_20"], t.loc[ep, "p_placebo_0_20"]
        col = EP_COL[ep]
        ax.plot([c0, c20], [yi, yi], color=col, lw=1.6, alpha=0.55, zorder=2,
                solid_capstyle="butt")
        ax.scatter(c0, yi, s=26, facecolor="white", edgecolor=col, lw=1.2, zorder=3)
        ax.scatter(c20, yi, s=30, facecolor=col, edgecolor="white", lw=0.5, zorder=4)
        lab_x, ha = max(c20, c0, 0.0) + 2.6, "left"
        ax.text(lab_x, yi, (f"{c20:+.1f}  (p = {p20:.3f})" if p20 < 0.05 else f"{c20:+.1f}  (p = {p20:.2f})").replace("-", "−"), va="center", ha=ha,
                fontsize=6.4,
                color=col, fontweight="semibold" if ep == "EP6" else "normal", path_effects=HALO, zorder=5)
    ax.axvline(0, color=C["ink"], lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{EV.set_index('episode_id').loc[e, 'short_label']}" for e in order])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(-25, 102)
    ax.set_xticks([-20, 0, 20, 40, 60, 80])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Brent cumulative abnormal return (%)")
    clean(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.set_title("A   Day-0 jump (open)  →  day-20 outcome (filled)", fontsize=7.6, pad=4)

    for yi, ep in zip(y, order):
        v = fr.loc[ep, "flow_chg_week1_pct"] if ep in fr.index else np.nan
        if np.isfinite(v):
            bx.barh(yi, v, height=0.5, color=EP_COL[ep], alpha=0.9 if ep == "EP6" else 0.6)
            bx.text(v + 3 if v < -40 else 2, yi, f"{v:.0f}%".replace("-", "−"), va="center",
                    ha="left", fontsize=6.4,
                    color="white" if v < -40 else C["ink"],
                    fontweight="semibold" if ep == "EP6" else "normal")
        else:
            bx.text(-100, yi, "no data (pre-2023)", va="center", ha="left", fontsize=6.0,
                    color=C["muted"], style="italic")
    noise = json.load(open(TAB / "summary.json"))["flow_noise_band_week1"]
    bx.axvspan(noise["p05_pct"], noise["p95_pct"], color="#D4DADF", alpha=0.7, lw=0, zorder=0)
    bx.axvline(0, color=C["ink"], lw=0.5)
    bx.set_xlim(-105, 32)
    bx.set_ylim(-0.6, len(order) - 0.4)
    bx.set_yticks([])
    bx.set_xticks([-100, -50, 0])
    bx.set_xlabel("Change against baseline (%)")
    clean(bx, grid="x")
    bx.spines["left"].set_visible(False)
    bx.set_title("B   Tanker transits, week 1", fontsize=7.6, pad=4)
    audit(fig, "fig6")
    save(fig, "fig6_episodes")
    plt.close(fig)


# ============================================================== Figure 7
def fig7():
    """Method validation on simulated data: CFAR detector and SAR-AIS fusion."""
    fig = plt.figure(figsize=(TEXTWIDTH, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.75], wspace=0.24,
                          left=0.035, right=0.985, top=0.885, bottom=0.135)
    ax = fig.add_subplot(gs[0])
    x, sea, truth = synth_scene(seed=3, n_ships=25)
    det = detect_vessels(x, sea)
    sc = score_detections(det, truth)
    crop = (slice(0, 1100), slice(0, 1100))
    ax.imshow(10 * np.log10(x[crop]), cmap="Greys_r", vmin=-32, vmax=-6, interpolation="nearest",
              rasterized=True)
    ax.contour(~sea[crop], levels=[0.5], colors=C["queue"], linewidths=0.7)
    tsel = truth[(truth.row < 1100) & (truth.col < 1100)]
    dsel = det[(det.row < 1100) & (det.col < 1100)]
    ax.scatter(tsel.col, tsel.row, s=60, facecolor="none", edgecolor="#56B4E9", lw=0.8)
    ax.scatter(dsel.col, dsel.row, s=14, marker="+", color=C["orange"], lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.5)
    ax.set_title(f"A   CFAR detector, synthetic scene", fontsize=7.6, pad=4)
    ax.text(0.5, -0.04, f"circles: true vessels · crosses: detections\n"
            f"precision {sc['precision']:.2f} · recall {sc['recall']:.2f} · 11 × 11 km",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=C["muted"],
            linespacing=1.3)
    h = [Line2D([], [], marker="o", ls="none", markersize=6, markerfacecolor="none",
                markeredgecolor="#56B4E9", markeredgewidth=0.8, label="true vessel"),
         Line2D([], [], marker="+", ls="none", markersize=5, markeredgecolor=C["orange"],
                markeredgewidth=0.8, label="detection")]

    bx = fig.add_subplot(gs[1])
    gate, v = 200.0, 12.0
    sim, cs = simulate_fusion_case(seed=0, gate_km=gate, speed_kn=v)
    fused = kalman_fuse(sim["ais_count"], sim["sar_count"], v * KN_TO_KM_PER_DAY / gate, cs)
    bx.plot(sim.index, sim["ais_count"], color="#9DB4C6", lw=0.5)
    bx.scatter(sim.index, snapshot_to_flow(sim["sar_count"], gate, v), s=4, color=C["ep2"], zorder=3,
               lw=0)
    bx.fill_between(fused.index, fused["flow_lo"], fused["flow_hi"], color=C["ep4"], alpha=0.16, lw=0)
    bx.plot(fused.index, fused["flow"], color=C["ep4"], lw=1.2)
    bx.plot(sim.index, sim["true_flow"], color=C["ink"], lw=0.8, ls=(0, (3, 1.6)))
    bx.axvline(cs, ymax=100 / 128, color=C["ink"], lw=0.5, ls=(0, (2, 2)))
    bx.set_ylim(0, 128)
    bx.set_yticks([0, 20, 40, 60, 80, 100])
    bx.set_ylabel("Transits per day")
    bx.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 9]))
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    clean(bx)
    bx.set_title("B   Radar–AIS fusion, simulated traffic", fontsize=7.6, pad=4)
    hb = [Line2D([], [], color=C["ink"], lw=0.8, ls=(0, (3, 1.6)), label="true flow"),
          Line2D([], [], color=C["ep4"], lw=1.2, label="fused index, 90% band"),
          Line2D([], [], marker="o", ls="none", markersize=2.6, color=C["ep2"], label="radar snapshots"),
          Line2D([], [], color="#9DB4C6", lw=0.8, label="AIS count (15% visible after break)")]
    bx.legend(handles=hb, loc="upper right", ncol=2, fontsize=6.0, columnspacing=1.0,
              handlelength=1.6, borderaxespad=0.2, frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)
    audit(fig, "fig7")
    save(fig, "fig7_methods")
    plt.close(fig)


if __name__ == "__main__":
    for k in sys.argv[1:] or ["4", "5", "6", "7"]:
        {"4": fig4, "5": fig5, "6": fig6, "7": fig7}[k]()
