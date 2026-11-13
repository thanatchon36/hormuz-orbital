"""
make_figures.py -- maps and charts for the research design.

    fig_regional_map()   Figure 1: strikes, energy infrastructure, chokepoints
    fig_hormuz_map()     Figure 2: Strait of Hormuz, SAR gate and queue zones
    fig_evidence()       Figure 3: transits vs Brent, and event-study paths
    fig_cfar_demo()      detector on a synthetic scene (README)
    fig_fusion_demo()    SAR-AIS fusion on simulated data (README)

Basemap: Natural Earth 1:10m vectors read from data/naturalearth (run
fetch_basemap.py once). Boundaries are shown for orientation only.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
NE = ROOT / "data" / "naturalearth"
REF = ROOT / "data" / "reference"
FIG = ROOT / "outputs" / "figures"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
PC = ccrs.PlateCarree()

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.fontsize": 7, "legend.frameon": True, "legend.framealpha": 0.94,
    "legend.edgecolor": "#c9c4ba", "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# palette (colour-blind safe, Okabe-Ito based)
SEA, DEEP, LAND, BORDER = "#e9f1f5", "#d5e4ed", "#f6f3ec", "#b8b0a2"
INK, MUTED = "#1f2a33", "#6d7780"
EPISODE_COLOR = {
    "2019-2022 incidents": "#0072B2",
    "2024 exchanges": "#E69F00",
    "June 2025 war": "#009E73",
    "2026 war": "#D55E00",
}
INFRA_MARKER = {"oil_terminal": "s", "oil_processing": "^", "oil_field": "^",
                "refinery": "^", "lng": "D", "gas_processing": "D"}
INFRA_LABEL = {"s": "Oil export terminal", "^": "Refinery or processing",
               "D": "LNG or gas processing"}
INFRA_COLOR = "#23323d"


# ------------------------------------------------------------------ helpers
def _layer(name, extent, pad=2.0):
    g = gpd.read_file(NE / f"{name}.geojson")
    x0, x1, y0, y1 = extent
    return g.cx[x0 - pad:x1 + pad, y0 - pad:y1 + pad]


def _basemap(ax, extent, *, minor_islands=False, deep=True):
    ax.set_extent(extent, crs=PC)
    ax.set_facecolor(SEA)
    if deep:
        try:
            b = _layer("ne_10m_bathymetry_K_200", extent)
            ax.add_geometries(b.geometry, PC, facecolor=DEEP, edgecolor="none", zorder=0.5)
        except Exception:
            pass
    land = _layer("ne_10m_land", extent)
    ax.add_geometries(land.geometry, PC, facecolor=LAND, edgecolor=BORDER,
                      linewidth=0.45, zorder=1)
    if minor_islands:
        mi = _layer("ne_10m_minor_islands", extent, pad=0.2)
        ax.add_geometries(mi.geometry, PC, facecolor=LAND, edgecolor=BORDER,
                          linewidth=0.4, zorder=1)
    lakes = _layer("ne_10m_lakes", extent)
    ax.add_geometries(lakes.geometry, PC, facecolor=SEA, edgecolor=BORDER,
                      linewidth=0.3, zorder=1.1)
    bl = _layer("ne_10m_admin_0_boundary_lines_land", extent)
    ax.add_geometries(bl.geometry, PC, facecolor="none", edgecolor=BORDER,
                      linewidth=0.5, zorder=1.2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#aab4bb",
                      alpha=0.6, linestyle=(0, (2, 3)), zorder=1.3)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5, "color": MUTED}
    for s in ax.spines.values():
        s.set_edgecolor("#7f8a92")
    return ax


def _text(ax, lon, lat, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    kw.setdefault("fontsize", 7); kw.setdefault("color", INK)
    kw.setdefault("zorder", 6)
    halo = kw.pop("halo", True)
    t = ax.text(lon, lat, s, transform=PC, **kw)
    if halo:
        import matplotlib.patheffects as pe
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white", alpha=0.85)])
    return t


def _scalebar(ax, lon, lat, km, *, label=None):
    """Horizontal scale bar starting at (lon, lat)."""
    dlon = km / (111.32 * np.cos(np.radians(lat)))
    y0, y1 = ax.get_extent(PC)[2:]
    tick = 0.012 * (y1 - y0)
    ax.plot([lon, lon + dlon], [lat, lat], transform=PC, color=INK, lw=1.6,
            solid_capstyle="butt", zorder=7)
    for x in (lon, lon + dlon):
        ax.plot([x, x], [lat, lat + tick], transform=PC, color=INK, lw=0.9, zorder=7)
    ax.text(lon + dlon / 2, lat + 1.6 * tick, label or f"{km:g} km", transform=PC,
            ha="center", va="bottom", fontsize=6.5, color=INK, zorder=7)


def _infra(ax, extent, *, size_scale=1.0, label_names=()):
    inf = pd.read_csv(REF / "energy_infrastructure.csv")
    x0, x1, y0, y1 = extent
    inf = inf[(inf.lon.between(x0, x1)) & (inf.lat.between(y0, y1))]
    for _, r in inf.iterrows():
        mk = INFRA_MARKER[r["type"]]
        size = (16 + 26 * np.sqrt(r["capacity_mboed"])) * size_scale
        ax.scatter(r["lon"], r["lat"], s=size, marker=mk, facecolor=INFRA_COLOR,
                   edgecolor="white", linewidth=0.6, transform=PC, zorder=4)
    return inf


def _strikes(ax, extent, *, size=30, jitter=None):
    st = pd.read_csv(REF / "strike_events_reference.csv", parse_dates=["date"])
    x0, x1, y0, y1 = extent
    st = st[(st.lon.between(x0, x1)) & (st.lat.between(y0, y1))].copy()
    for ep, col in EPISODE_COLOR.items():
        s = st[st.episode == ep]
        hi, ve = s[s.confidence == "high"], s[s.confidence != "high"]
        ax.scatter(hi.lon, hi.lat, s=size, marker="o", facecolor=col, edgecolor="white",
                   linewidth=0.5, alpha=0.92, transform=PC, zorder=5)
        ax.scatter(ve.lon, ve.lat, s=size, marker="o", facecolor="none", edgecolor=col,
                   linewidth=1.3, transform=PC, zorder=5)
    return st


def _legend_handles(infra=True):
    h = []
    for ep, col in EPISODE_COLOR.items():
        hollow = ep == "2026 war"
        h.append(Line2D([], [], marker="o", linestyle="none", markersize=5.5,
                        markerfacecolor="none" if hollow else col,
                        markeredgecolor=col if hollow else "white",
                        markeredgewidth=1.3 if hollow else 0.5,
                        label=f"Strike site, {ep.replace('2019-2022', '2019–2022')}"
                              + (" (to verify)" if hollow else "")))
    if infra:
        for mk, lab in INFRA_LABEL.items():
            h.append(Line2D([], [], marker=mk, linestyle="none", markersize=5.5,
                            markerfacecolor=INFRA_COLOR, markeredgecolor="white",
                            markeredgewidth=0.6, label=lab))
    return h


# ------------------------------------------------------------ Figure 1: region
def _legend_below(fig, handles, y, ncol=3):
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                     ncol=ncol, frameon=False, columnspacing=1.6, handletextpad=0.5,
                     labelspacing=0.5, fontsize=7)
    return leg


def fig_regional_map(out=FIG / "fig1_regional_map.png"):
    extent = CFG["map_extent"]["regional"]
    hz = CFG["map_extent"]["hormuz"]
    fig = plt.figure(figsize=(7.6, 7.9))
    ax = fig.add_axes([0.06, 0.165, 0.88, 0.795], projection=ccrs.Mercator())
    _basemap(ax, extent)

    # Hormuz bypass pipelines (schematic polylines)
    pipes = {
        "East–West pipeline": [(49.68, 25.93), (46.0, 24.75), (42.0, 24.35), (38.10, 24.02)],
        "Habshan–Fujairah": [(53.63, 23.85), (55.2, 24.6), (56.36, 25.17)],
        "Goreh–Jask": [(50.42, 29.86), (52.9, 28.2), (55.5, 27.35), (56.9, 27.3),
                       (57.3, 26.6), (57.77, 25.64)],
    }
    for pts in pipes.values():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, transform=PC, color="#5b6770", lw=1.0,
                linestyle=(0, (4, 2)), zorder=3)
    _text(ax, 43.6, 25.15, "East–West pipeline", fontsize=6.2, color="#4d5961", rotation=6)

    _infra(ax, extent, size_scale=0.8)
    _strikes(ax, extent, size=24)

    # chokepoints
    for name, lon, lat, dx, dy, ha in [
            ("Strait of Hormuz", 56.45, 26.55, 0.75, 1.75, "center"),
            ("Bab el-Mandeb", 43.33, 12.58, 0.7, -0.35, "left"),
            ("Suez Canal", 32.35, 30.5, -0.45, 0.2, "right")]:
        ax.scatter(lon, lat, s=120, marker="*", facecolor="#fdd835", edgecolor=INK,
                   linewidth=0.6, transform=PC, zorder=6)
        _text(ax, lon + dx, lat + dy, name, ha=ha, fontsize=7.4, fontweight="bold")

    # extent of Figure 2
    ax.add_patch(Rectangle((hz[0], hz[2]), hz[1] - hz[0], hz[3] - hz[2], transform=PC,
                           fill=False, edgecolor=INK, lw=0.9, zorder=6))
    _text(ax, hz[1] + 0.25, hz[3] - 0.3, "Fig. 2", ha="left", fontsize=6.5, color=INK,
          style="italic")

    # country and sea labels
    for name, lon, lat in [("IRAN", 55.6, 31.9), ("IRAQ", 43.3, 32.2), ("SAUDI ARABIA", 45.0, 22.6),
                           ("OMAN", 57.2, 21.3), ("UAE", 54.6, 23.0), ("YEMEN", 46.5, 15.6),
                           ("EGYPT", 31.2, 27.2), ("TURKEY", 36.5, 38.9), ("SYRIA", 38.4, 35.1),
                           ("JORDAN", 37.0, 30.9), ("KUWAIT", 46.45, 29.0),
                           ("PAKISTAN", 61.3, 28.6), ("AFGHANISTAN", 61.3, 33.2),
                           ("TURKMENISTAN", 58.6, 39.3)]:
        _text(ax, lon, lat, name, fontsize=6.3, color="#8a8274", halo=False, fontweight="bold")
    for name, lon, lat, rot in [("Persian Gulf", 51.35, 27.35, -35), ("Gulf of Oman", 60.3, 24.5, -12),
                                ("Arabian Sea", 59.6, 17.5, 0), ("Red Sea", 37.6, 20.6, -58),
                                ("Mediterranean Sea", 31.7, 33.7, 0), ("Caspian\nSea", 51.0, 38.6, 0),
                                ("Gulf of Aden", 47.6, 12.6, 12)]:
        _text(ax, lon, lat, name, fontsize=6.6, color="#4f7f9c", style="italic",
              rotation=rot, halo=False)

    # place labels: (name, lon, lat, dx, dy, ha)
    for name, lon, lat, dx, dy, ha in [
            ("Tehran", 51.40, 35.70, 0.35, 0.42, "left"), ("Natanz", 51.73, 33.72, 0.4, 0.1, "left"),
            ("Fordow", 50.99, 34.88, -0.35, 0.2, "right"), ("Isfahan", 51.67, 32.6, 0.45, -0.3, "left"),
            ("Arak", 49.24, 34.37, 0.0, -0.45, "center"), ("Kermanshah", 47.07, 34.31, -0.4, 0.0, "right"),
            ("Tabriz", 46.24, 38.13, 0.4, 0.0, "left"), ("Kharg I.", 50.31, 29.24, 0.45, 0.3, "left"),
            ("South Pars", 52.61, 27.48, -0.1, 0.62, "center"),
            ("Ras Laffan (Qatar)", 51.56, 25.91, 0.45, 0.33, "left"), ("Ras Tanura", 50.18, 26.70, -0.45, 0.3, "right"),
            ("Abqaiq", 49.68, 25.93, -0.35, -0.42, "right"), ("Khurais", 48.19, 25.07, -0.35, 0.0, "right"),
            ("Yanbu", 38.10, 24.02, 0.0, -0.62, "center"), ("Bahrain", 50.61, 26.21, 0.05, -0.6, "center"),
            ("Basra terminals", 48.81, 29.78, 0.4, 0.42, "left"), ("Jeddah", 39.14, 21.71, 0.35, 0.0, "left"),
            ("Ain al-Asad", 42.44, 33.79, -0.35, 0.1, "right"), ("Erbil", 43.96, 36.24, 0.4, 0.0, "left"),
            ("Abu Dhabi", 54.50, 24.35, -0.35, -0.5, "center")]:
        _text(ax, lon + dx, lat + dy, name, ha=ha, fontsize=6.3)

    # inset: Israel, where sites are too close to separate at regional scale
    ins_ext = [34.1, 35.8, 30.85, 33.2]
    ax.add_patch(Rectangle((ins_ext[0], ins_ext[2]), ins_ext[1] - ins_ext[0],
                           ins_ext[3] - ins_ext[2], transform=PC, fill=False,
                           edgecolor=INK, lw=0.8, zorder=6))
    axi = ax.inset_axes([0.012, 0.095, 0.20, 0.37], projection=ccrs.Mercator())
    axi.set_extent(ins_ext, crs=PC)
    axi.set_facecolor(SEA)
    land = _layer("ne_10m_land", ins_ext, pad=0.5)
    axi.add_geometries(land.geometry, PC, facecolor=LAND, edgecolor=BORDER, linewidth=0.45)
    bl = _layer("ne_10m_admin_0_boundary_lines_land", ins_ext, pad=0.5)
    axi.add_geometries(bl.geometry, PC, facecolor="none", edgecolor=BORDER, linewidth=0.45)
    _infra(axi, ins_ext, size_scale=1.5)
    _strikes(axi, ins_ext, size=30)
    for name, lon, lat, dx, dy, ha in [
            ("Haifa", 35.05, 32.80, 0.09, 0.07, "left"), ("Tel Aviv", 34.80, 32.08, -0.08, 0.06, "right"),
            ("Rehovot", 34.81, 31.91, 0.09, 0.0, "left"),
            ("Beersheba", 34.80, 31.26, -0.08, 0.0, "right"), ("Nevatim", 35.01, 31.21, 0.08, -0.02, "left")]:
        _text(axi, lon + dx, lat + dy, name, ha=ha, fontsize=5.8)
    axi.set_title("Israel (enlarged)", fontsize=6.6, fontweight="bold", loc="center", pad=2.5)
    for sp in axi.spines.values():
        sp.set_edgecolor(INK); sp.set_linewidth(0.8)
    _scalebar(ax, 31.0, 11.9, 500)
    handles = _legend_handles() + [
        Line2D([], [], marker="*", linestyle="none", markersize=9, markerfacecolor="#fdd835",
               markeredgecolor=INK, markeredgewidth=0.6, label="Maritime chokepoint"),
        Line2D([], [], color="#5b6770", lw=1.0, linestyle=(0, (4, 2)),
               label="Hormuz bypass pipeline (schematic)"),
        Patch(facecolor=DEEP, edgecolor="none", label="Water deeper than 200 m")]
    _legend_below(fig, handles, 0.062)
    ax.set_title("Strike sites and energy infrastructure around the Gulf, 2019–2026")
    fig.text(0.06, 0.012,
             "Strike sites are an illustrative reference layer compiled by the author; 2026 sites are "
             "unverified. The analysis uses ACLED records.\nInfrastructure marker area scales with "
             "indicative capacity. Basemap: Natural Earth. Boundaries are for orientation only.",
             fontsize=6.2, color=MUTED, va="bottom", linespacing=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------------------------------------------------ Figure 2: Hormuz
def fig_hormuz_map(out=FIG / "fig2_hormuz_map.png"):
    extent = CFG["map_extent"]["hormuz"]
    fig = plt.figure(figsize=(7.6, 7.15))
    ax = fig.add_axes([0.06, 0.185, 0.88, 0.77], projection=ccrs.Mercator())
    _basemap(ax, extent, minor_islands=True)

    # schematic route
    route = np.array(CFG["route_schematic"])
    ax.plot(route[:, 0], route[:, 1], transform=PC, color="#34495e", lw=1.3,
            linestyle=(0, (5, 2.5)), zorder=3)
    _text(ax, 58.02, 25.13, "main route\n(schematic)", fontsize=6.3, color="#34495e",
          style="italic", ha="left")

    # areas of interest
    aoi_style = {"gate": dict(edgecolor="#b3261e", facecolor="#b3261e22", lw=1.4, ls="-"),
                 "queue_oman": dict(edgecolor="#5e35b1", facecolor="#5e35b114", lw=1.1, ls=(0, (3, 2))),
                 "queue_gulf": dict(edgecolor="#5e35b1", facecolor="#5e35b114", lw=1.1, ls=(0, (3, 2)))}
    for key, st in aoi_style.items():
        a = CFG["aoi"][key]
        ax.add_patch(Rectangle((a["lon_min"], a["lat_min"]), a["lon_max"] - a["lon_min"],
                               a["lat_max"] - a["lat_min"], transform=PC, zorder=2.5,
                               edgecolor=st["edgecolor"], facecolor=st["facecolor"],
                               linewidth=st["lw"], linestyle=st["ls"]))
    g = CFG["aoi"]["gate"]
    _text(ax, g["lon_max"] + 0.05, g["lat_min"] + 0.1, "SAR transit gate", ha="left",
          fontsize=7, color="#8c1d18", fontweight="bold")
    q = CFG["aoi"]["queue_oman"]
    _text(ax, q["lon_min"] + 0.05, q["lat_min"] + 0.1, "queue zone (Gulf of Oman)", ha="left",
          fontsize=6.6, color="#4a2a8f")
    q = CFG["aoi"]["queue_gulf"]
    _text(ax, q["lon_min"] + 0.05, q["lat_min"] + 0.1, "queue zone (inside the Gulf)", ha="left",
          fontsize=6.6, color="#4a2a8f")

    _infra(ax, extent, size_scale=1.2)
    _strikes(ax, extent, size=42)

    for name, lon, lat, dx, dy, ha in [
            ("Bandar Abbas", 56.27, 27.19, 0.0, 0.14, "center"), ("Qeshm", 55.85, 26.82, 0.0, 0.0, "center"),
            ("Larak", 56.36, 26.86, 0.17, 0.0, "left"), ("Hormuz I.", 56.46, 27.06, 0.16, 0.03, "left"),
            ("Khasab", 56.25, 26.18, 0.09, 0.0, "left"),
            ("Dubai", 55.27, 25.20, 0.0, -0.13, "center"), ("Fujairah", 56.34, 25.12, -0.12, -0.08, "right"),
            ("Sohar", 56.62, 24.49, -0.12, 0.0, "right"), ("Jask", 57.77, 25.64, 0.0, 0.14, "center"),
            ("Bandar Lengeh", 54.88, 26.56, 0.0, 0.13, "center"), ("Abu Musa", 55.03, 25.87, 0.0, -0.11, "center"),
            ("Greater Tunb", 55.30, 26.26, 0.0, 0.11, "center"),
            ("tankers attacked,\nJune 2019", 57.40, 25.45, 0.12, -0.07, "left"),
            ("Musaffah", 54.50, 24.35, 0.1, 0.0, "left")]:
        _text(ax, lon + dx, lat + dy, name, ha=ha, fontsize=6.6)
    for name, lon, lat in [("IRAN", 57.6, 27.3), ("OMAN", 57.0, 24.42),
                           ("UNITED ARAB\nEMIRATES", 55.35, 24.62), ("MUSANDAM\n(OMAN)", 56.22, 25.98)]:
        _text(ax, lon, lat, name, fontsize=6.4, color="#8a8274", halo=False, fontweight="bold")
    _text(ax, 54.62, 26.3, "Persian Gulf", fontsize=7.2, color="#4f7f9c", style="italic", halo=False)
    _text(ax, 58.05, 24.75, "Gulf of Oman", fontsize=7.2, color="#4f7f9c", style="italic", halo=False)
    _text(ax, 56.55, 26.715, "Strait of Hormuz", fontsize=7.6, color=INK, fontweight="bold")

    _scalebar(ax, 54.4, 24.48, 50)
    handles = [
        Patch(facecolor="#b3261e22", edgecolor="#b3261e", lw=1.4, label="SAR transit gate (flow)"),
        Patch(facecolor="#5e35b114", edgecolor="#5e35b1", lw=1.1, linestyle=(0, (3, 2)),
              label="Queue zones (waiting stock)"),
        Line2D([], [], color="#34495e", lw=1.3, linestyle=(0, (5, 2.5)), label="Main route (schematic)"),
        Patch(facecolor=DEEP, edgecolor="none", label="Water deeper than 200 m"),
    ] + _legend_handles()
    _legend_below(fig, handles, 0.06)
    ax.set_title("Strait of Hormuz: satellite counting design")
    fig.text(0.06, 0.012,
             "Gate, queue zones and route are design placeholders to be refined against an AIS density "
             "raster. Not for navigation.\nStrike sites are illustrative; 2026 sites are unverified. "
             "Basemap: Natural Earth.",
             fontsize=6.2, color=MUTED, va="bottom", linespacing=1.4)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------- Figure 3: evidence charts
EP_STYLE = {   # colour, line width, dash, y position of the end label
    "EP6": ("#D55E00", 2.4, "-", None),
    "EP5": ("#009E73", 1.5, "-", 8.6),
    "EP4": ("#E69F00", 1.3, "-", 3.9),
    "EP1": ("#0072B2", 1.5, "-", -0.8),
    "EP3": ("#7f8a92", 1.1, (0, (1, 1.5)), -11.2),
    "EP2": ("#7f8a92", 1.1, (0, (4, 2)), -17.0),
}


def fig_evidence(out=FIG / "fig3_preliminary_evidence.png"):
    """Three panels from real data: Hormuz tanker transits (IMF PortWatch),
    Brent spot (EIA), and event-study paths for the six episode onsets."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_sources import load_oil_prices, load_portwatch, load_reference
    from event_study import placebo_cars, run_event_table

    px, pw, ev = load_oil_prices(), load_portwatch("hormuz"), load_reference("episode_onsets")
    es = CFG["event_study"]
    est, win = tuple(es["estimation_window"]), tuple(es["event_window"])
    table, paths = run_event_table(px["brent"], ev, est=est, win=win,
                                   placebo_start=es["placebo_start"])
    # placebo band for CAR(0, k)
    ks = np.arange(0, win[1] + 1)
    band = np.array([np.quantile(placebo_cars(px["brent"], est=est, window=(0, int(k)),
                                              start=es["placebo_start"],
                                              exclude=list(ev["day0"])).values, [0.05, 0.95])
                     for k in ks])

    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.75], hspace=0.34,
                          left=0.09, right=0.965, top=0.955, bottom=0.10)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), None, fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    t0, t1 = pd.Timestamp("2023-01-01"), px.index.max()
    war = pd.Timestamp("2026-02-28")

    # (a) transits
    tank = pw["n_tanker"]
    ax1.plot(tank.index, tank.values, color="#9fb7c9", lw=0.5, zorder=2)
    ax1.plot(tank.index, tank.rolling(7, center=True).mean(), color="#0b3c5d", lw=1.4, zorder=3)
    ax1.set_ylabel("Tanker transits per day")
    ax1.set_ylim(0, 92)
    ax1.set_title("(a) Hormuz tanker transits per day, AIS-derived (IMF PortWatch)")
    # (b) Brent
    b = px["brent"].dropna().loc[t0:]
    ax2.plot(b.index, b.values, color="#D55E00", lw=1.2, zorder=3)
    ax2.set_ylabel("USD per barrel")
    ax2.set_ylim(50, 148)
    ax2.set_title("(b) Brent spot price (EIA)")
    pk = b.idxmax()
    ax2.annotate(f"{b.max():.2f} on {pk:%-d %b %Y}", xy=(pk, b.max()), xytext=(-78, -3),
                 textcoords="offset points", fontsize=6.8, color=INK, va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    marks = [("2024-04-13", "Apr 2024"), ("2024-10-01", "Oct 2024"),
             ("2025-06-13", "Jun 2025"), ("2026-02-28", "28 Feb 2026")]
    for ax in (ax1, ax2):
        ax.axvspan(war, t1, color="#D55E00", alpha=0.06, zorder=0, lw=0)
        for d, lab in marks:
            ax.axvline(pd.Timestamp(d), color=INK, lw=0.7, linestyle=(0, (2, 2)), zorder=1)
        ax.set_xlim(t0, t1)
        ax.grid(axis="y", color="#d9dee2", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for d, lab in marks:
        ax1.text(pd.Timestamp(d) - pd.Timedelta(days=9), 90, lab, fontsize=6.6, color=INK,
                 ha="right", va="top")
    ax1.tick_params(labelbottom=False)

    # (c) event-study paths
    ax3.fill_between(ks, band[:, 0], band[:, 1], color="#c9cfd4", alpha=0.55, lw=0, zorder=1,
                     label="5th–95th percentile of placebo dates")
    ax3.axhline(0, color=INK, lw=0.6, zorder=2)
    ax3.axvline(0, color=INK, lw=0.7, linestyle=(0, (2, 2)), zorder=2)
    labels = dict(zip(ev["episode_id"], ev["short_label"]))
    for ep, (col, lw, ls, ylab) in EP_STYLE.items():
        p = paths[paths["episode_id"] == ep]
        ax3.plot(p["tau"], p["car"], color=col, lw=lw, linestyle=ls, zorder=4 if ep == "EP6" else 3)
        last = p.iloc[-1]
        ax3.text(last["tau"] + 0.35, last["car"] if ylab is None else ylab, labels[ep], color=col, fontsize=7,
                 va="center", ha="left", fontweight="bold" if ep == "EP6" else "normal")
    ax3.set_xlim(win[0], win[1] + 3.6)
    ax3.set_ylim(-20, 58)
    ax3.set_xticks(np.arange(-5, 21, 5))
    ax3.set_xlabel("Trading days from event day 0")
    ax3.set_ylabel("Cumulative abnormal return, log points (%)")
    ax3.set_title("(c) Brent around six episode onsets: only the episode that stopped the tankers kept rising")
    ax3.grid(axis="y", color="#d9dee2", lw=0.5)
    ax3.set_axisbelow(True)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    ax3.legend(loc="upper left", frameon=False, fontsize=7)
    fig.text(0.09, 0.012,
             f"Sources: IMF PortWatch daily chokepoint data to {pw.index.max():%-d %b %Y} (snapshot of 28 Oct 2026); "
             f"EIA Brent spot to {px.index.max():%-d %b %Y}. In (a) the thin line is daily, the thick line a 7-day mean.\n"
             f"Constant-mean model, estimation window "
             f"[{est[0]}, {est[1]}] trading days. Placebo dates: {es['placebo_start'][:4]} onward, event windows excluded.",
             fontsize=6.2, color=MUTED, va="bottom", linespacing=1.4)
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------------------------------------- README figures (simulated)
def fig_cfar_demo(out=FIG / "demo_cfar_synthetic.png", seed=3):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s1_cfar import detect_vessels, score_detections, synth_scene
    x, sea, truth = synth_scene(seed=seed, n_ships=25)
    det = detect_vessels(x, sea)
    sc = score_detections(det, truth)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.imshow(10 * np.log10(x), cmap="gray", vmin=-32, vmax=-8, interpolation="nearest")
    ax.contour(~sea, levels=[0.5], colors="#5e35b1", linewidths=0.8)
    ax.scatter(truth["col"], truth["row"], s=110, facecolor="none", edgecolor="#56B4E9", lw=1.0,
               label=f"true vessels ({len(truth)})")
    ax.scatter(det["col"], det["row"], s=22, marker="+", color="#D55E00", lw=1.0,
               label=f"detections ({len(det)})")
    ax.set_title(f"CFAR on a synthetic scene: precision {sc['precision']:.2f}, recall {sc['recall']:.2f}")
    ax.set_xlabel("range pixel (10 m)"); ax.set_ylabel("azimuth pixel (10 m)")
    ax.legend(loc="lower left")
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_fusion_demo(out=FIG / "demo_fusion_simulated.png", seed=0):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flow_index import KN_TO_KM_PER_DAY, kalman_fuse, simulate_fusion_case, snapshot_to_flow
    gate, v = 200.0, 12.0
    sim, cs = simulate_fusion_case(seed=seed, gate_km=gate, speed_kn=v)
    fused = kalman_fuse(sim["ais_count"], sim["sar_count"], v * KN_TO_KM_PER_DAY / gate, cs)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(sim.index, sim["ais_count"], color="#9fb7c9", lw=0.6, label="AIS count (under-reports after the break)")
    ax.scatter(sim.index, snapshot_to_flow(sim["sar_count"], gate, v), s=7, color="#E69F00",
               label="SAR snapshots (Little's law)", zorder=3)
    ax.fill_between(fused.index, fused["flow_lo"], fused["flow_hi"], color="#D55E00", alpha=0.18, lw=0)
    ax.plot(fused.index, fused["flow"], color="#D55E00", lw=1.5, label="Fused index, 90% band")
    ax.plot(sim.index, sim["true_flow"], color=INK, lw=1.0, linestyle=(0, (4, 2)), label="True flow (simulated)")
    ax.axvline(cs, color=INK, lw=0.7, linestyle=(0, (2, 2)))
    ax.set_ylabel("Transits per day"); ax.set_ylim(0, None)
    ax.set_title("SAR–AIS fusion on SIMULATED data")
    ax.legend(loc="upper right", ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out); plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig_regional_map, fig_hormuz_map, fig_evidence, fig_cfar_demo, fig_fusion_demo):
        print(f())
