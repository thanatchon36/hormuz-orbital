"""fig_maps.py -- Figures 1-3 (maps). Run: python fig_maps.py [1 2 3]"""
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from figlib import (C, EPISODES, HALO, INFRA_LABEL, INFRA_MARKER, PC, REF, REPO, TEXTWIDTH,
                    Placer, audit, basemap, save, scalebar, style)

style()
CFG = yaml.safe_load(open(REPO / "config.yaml"))
INFRA = pd.read_csv(REF / "energy_infrastructure.csv")
STRIKES = pd.read_csv(REF / "strike_events_reference.csv", parse_dates=["date"])

REGION = [30.0, 62.5, 11.0, 40.5]
GULF = [46.6, 58.9, 23.2, 30.9]
ISRAEL = [34.1, 35.8, 30.85, 33.2]
HORMUZ = CFG["map_extent"]["hormuz"]

PIPES = {
    "East–West pipeline": [(49.68, 25.93), (46.0, 24.75), (42.0, 24.35), (38.10, 24.02)],
    "Habshan–Fujairah": [(53.63, 23.85), (55.2, 24.6), (56.36, 25.17)],
    "Goreh–Jask": [(50.42, 29.86), (52.9, 28.2), (55.5, 27.35), (56.9, 27.3),
                   (57.3, 26.6), (57.77, 25.64)],
}


def inside(df, ext):
    return df[df.lon.between(ext[0], ext[1]) & df.lat.between(ext[2], ext[3])]


def draw_infra(ax, ext, placer=None, base=8.0, gain=9.0):
    for _, r in inside(INFRA, ext).iterrows():
        s = base + gain * np.sqrt(r["capacity_mboed"])
        ax.scatter(r["lon"], r["lat"], s=s, marker=INFRA_MARKER[r["type"]], facecolor=C["infra"],
                   edgecolor="white", linewidth=0.5, transform=PC, zorder=4)
        if placer is not None:
            placer.add_marker(r["lon"], r["lat"], s * 1.25)


def draw_strikes(ax, ext, placer=None, s=17.0):
    for key, _, col in EPISODES:
        d = inside(STRIKES[STRIKES.episode == key], ext)
        solid, hollow = d[d.confidence == "high"], d[d.confidence != "high"]
        ax.scatter(solid.lon, solid.lat, s=s, marker="o", facecolor=col, edgecolor="white",
                   linewidth=0.45, transform=PC, zorder=5)
        ax.scatter(hollow.lon, hollow.lat, s=s * 1.25, marker="o", facecolor="none",
                   edgecolor=col, linewidth=1.15, transform=PC, zorder=5.2)
        if placer is not None:
            for _, r in d.iterrows():
                placer.add_marker(r["lon"], r["lat"], s * 1.7)


def draw_pipes(ax, placer=None, names=None):
    for name, pts in PIPES.items():
        if names and name not in names:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, transform=PC, color="#5B6770", lw=0.9, linestyle=(0, (4, 2)), zorder=3)
        if placer is not None:
            placer.add_line(xs, ys, width_pt=1.0)


def legend_handles(strikes=True, infra=True):
    h = []
    if strikes:
        for key, lab, col in EPISODES:
            hollow = key == "2026 war"
            h.append(Line2D([], [], marker="o", linestyle="none", markersize=4.6,
                            markerfacecolor="none" if hollow else col,
                            markeredgecolor=col if hollow else "white",
                            markeredgewidth=1.15 if hollow else 0.45,
                            label=f"Strike site, {lab}"))
    if infra:
        for mk, lab in INFRA_LABEL:
            h.append(Line2D([], [], marker=mk, linestyle="none", markersize=4.8,
                            markerfacecolor=C["infra"], markeredgecolor="white",
                            markeredgewidth=0.5, label=lab))
    return h


STAR = dict(marker="*", s=95, facecolor=C["gold"], edgecolor=C["ink"], linewidth=0.55, zorder=6)
H_STAR = Line2D([], [], marker="*", linestyle="none", markersize=8.5, markerfacecolor=C["gold"],
                markeredgecolor=C["ink"], markeredgewidth=0.55, label="Maritime chokepoint")
H_PIPE = Line2D([], [], color="#5B6770", lw=0.9, linestyle=(0, (4, 2)),
                label="Hormuz bypass pipeline (schematic)")
H_DEEP = Patch(facecolor=C["deep"], edgecolor="none", label="Water deeper than 200 m")
COUNTRY = dict(fontsize=6.3, color=C["country"], fontweight="semibold")
SEA = dict(fontsize=6.7, color=C["seatext"], style="italic")


# ============================================================== Figure 1
def fig1():
    fig = plt.figure(figsize=(TEXTWIDTH, 7.22))
    ax = fig.add_axes([0.068, 0.138, 0.922, 0.856], projection=ccrs.Mercator())
    basemap(ax, REGION, xticks=np.arange(35, 61, 5), yticks=np.arange(15, 41, 5))

    draw_pipes(ax)
    draw_infra(ax, REGION)
    draw_strikes(ax, REGION)
    chokes = [("Strait of Hormuz", 56.45, 26.55), ("Bab el-Mandeb", 43.33, 12.58),
              ("Suez Canal", 32.35, 30.50)]
    for _, lon, lat in chokes:
        ax.scatter(lon, lat, transform=PC, **STAR)
    for ext in (GULF, ISRAEL):
        ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], transform=PC,
                               fill=False, edgecolor=C["ink"], lw=0.8, zorder=6))

    # inset: Israel
    axi = ax.inset_axes([0.012, 0.075, 0.225, 0.40], projection=ccrs.Mercator())
    basemap(axi, ISRAEL)
    for sp in axi.spines.values():
        sp.set_edgecolor(C["ink"]); sp.set_linewidth(0.8)

    P = Placer(ax)
    P.add_box(axi.get_window_extent(P.r).expanded(1.03, 1.06), "inset")
    draw_infra_reg = inside(INFRA, REGION)
    for _, r in draw_infra_reg.iterrows():
        P.add_marker(r["lon"], r["lat"], (8 + 9 * np.sqrt(r["capacity_mboed"])) * 1.25)
    for _, r in inside(STRIKES, REGION).iterrows():
        P.add_marker(r["lon"], r["lat"], 17 * 1.7)
    for _, lon, lat in chokes:
        P.add_marker(lon, lat, 110)
    for pts in PIPES.values():
        P.add_line(*zip(*pts), width_pt=1.0)
    for ext in (GULF, ISRAEL):
        P.add_rect_outline(ext[0], ext[1], ext[2], ext[3])

    # fixed background typography
    for s, lon, lat in [("TURKEY", 36.4, 38.9), ("SYRIA", 38.5, 35.0), ("IRAQ", 43.0, 32.35),
                        ("IRAN", 56.0, 32.4), ("JORDAN", 37.0, 30.75), ("EGYPT", 31.3, 27.2),
                        ("SAUDI ARABIA", 44.6, 22.3), ("YEMEN", 46.9, 15.6), ("OMAN", 57.1, 20.6),
                        ("PAKISTAN", 60.9, 29.3), ("AFGHANISTAN", 60.2, 33.6),
                        ("TURKMENISTAN", 58.6, 39.3)]:
        P.fixed(lon, lat, s, **COUNTRY)
    for s, lon, lat, rot in [("Mediterranean Sea", 32.4, 34.2, 0), ("Red Sea", 38.75, 18.9, -58),
                             ("Arabian Sea", 60.0, 17.4, 0), ("Gulf of Oman", 60.75, 24.25, -10),
                             ("Caspian\nSea", 51.0, 38.6, 0), ("Gulf of Aden", 47.9, 12.55, 11)]:
        P.fixed(lon, lat, s, rotation=rot, linespacing=1.05, **SEA)
    P.fixed(GULF[0] + 0.1, GULF[3] + 0.22, "Fig. 2", ha="left", va="bottom", fontsize=6.5,
            style="italic")
    scalebar(ax, 37.6, 11.75, 500, placer=P)

    # Hormuz: label outside the Fig. 2 frame, tied to its star by a leader
    t = P.fixed(59.3, 26.95, "Strait of\nHormuz", ha="left", fontsize=7.1, fontweight="semibold",
                linespacing=1.1)
    ax.plot([56.72, 59.15], [26.62, 26.9], transform=PC, color=C["ink"], lw=0.5, zorder=6.5)
    for s, lon, lat in chokes[1:]:
        P.place(lon, lat, s, fontsize=7.1, fontweight="semibold", dists=(7, 10, 14, 19, 25, 32),
                leader_from=99, prefer=("E", "W", "NE", "NW", "SE", "SW", "N", "S"))
    for s, lon, lat in [("Tehran", 51.40, 35.70), ("Fordow", 50.99, 34.88), ("Natanz", 51.73, 33.72),
                        ("Isfahan", 51.75, 32.66), ("Arak", 49.24, 34.37), ("Kermanshah", 47.07, 34.31),
                        ("Tabriz", 46.24, 38.13), ("Erbil", 43.96, 36.24), ("Ain al-Asad", 42.44, 33.79),
                        ("Yanbu", 38.10, 24.02), ("Jeddah", 39.14, 21.71)]:
        P.place(lon, lat, s)

    # inset content and labels
    draw_infra(axi, ISRAEL, base=10, gain=14)
    draw_strikes(axi, ISRAEL, s=20)
    Pi = Placer(axi, margin_pt=1.0)
    for _, r in inside(INFRA, ISRAEL).iterrows():
        Pi.add_marker(r["lon"], r["lat"], (10 + 14 * np.sqrt(r["capacity_mboed"])) * 1.25)
    for _, r in inside(STRIKES, ISRAEL).iterrows():
        Pi.add_marker(r["lon"], r["lat"], 20 * 1.7)
    Pi.fixed(34.42, 33.07, "Israel", fontsize=6.6, fontweight="semibold", ha="center")
    for s, lon, lat in [("Haifa", 35.03, 32.80), ("Tel Aviv", 34.80, 32.08), ("Rehovot", 34.81, 31.91),
                        ("Ashdod", 34.67, 31.83), ("Nevatim", 35.01, 31.21),
                        ("Beersheba", 34.80, 31.26), ("Leviathan", 34.80, 32.60)]:
        Pi.place(lon, lat, s, fontsize=5.9, dists=(4.5, 7, 10, 14, 19, 25), leader_from=9)

    fig.legend(handles=legend_handles() + [H_STAR, H_PIPE, H_DEEP], loc="lower center",
               bbox_to_anchor=(0.53, 0.004), ncol=3, columnspacing=1.5, handletextpad=0.5,
               labelspacing=0.5, fontsize=6.9)
    print("placer failures:", P.failed + Pi.failed)
    audit(fig, "fig1", symbols=[b for b, t in P.boxes if t == "symbol"])
    save(fig, "fig1_region")
    plt.close(fig)


# ============================================================== Figure 2
def fig2():
    fig = plt.figure(figsize=(TEXTWIDTH, 5.25))
    ax = fig.add_axes([0.068, 0.165, 0.922, 0.825], projection=ccrs.Mercator())
    basemap(ax, GULF, minor_islands=True, xticks=np.arange(48, 59, 2), yticks=np.arange(24, 31, 2))

    route = np.array(CFG["route_schematic"])
    ax.plot(route[:, 0], route[:, 1], transform=PC, color="#34495E", lw=1.0,
            linestyle=(0, (1.2, 1.6)), zorder=3)
    draw_pipes(ax)
    draw_infra(ax, GULF, base=12, gain=15)
    draw_strikes(ax, GULF, s=24)
    ax.scatter(56.45, 26.55, transform=PC, **STAR)
    ax.add_patch(Rectangle((HORMUZ[0], HORMUZ[2]), HORMUZ[1] - HORMUZ[0], HORMUZ[3] - HORMUZ[2],
                           transform=PC, fill=False, edgecolor=C["ink"], lw=0.8, zorder=6))

    P = Placer(ax)
    for _, r in inside(INFRA, GULF).iterrows():
        P.add_marker(r["lon"], r["lat"], (12 + 15 * np.sqrt(r["capacity_mboed"])) * 1.25)
    for _, r in inside(STRIKES, GULF).iterrows():
        P.add_marker(r["lon"], r["lat"], 24 * 1.7)
    P.add_marker(56.45, 26.55, 110)
    for pts in PIPES.values():
        P.add_line(*zip(*pts), width_pt=1.0)
    P.add_line(route[:, 0], route[:, 1], width_pt=1.0)
    P.add_rect_outline(HORMUZ[0], HORMUZ[1], HORMUZ[2], HORMUZ[3])

    for s, lon, lat in [("IRAN", 54.4, 29.7), ("SAUDI ARABIA", 49.0, 24.2), ("IRAQ", 47.05, 30.55),
                        ("UNITED ARAB EMIRATES", 54.55, 23.55), ("OMAN", 57.55, 23.75)]:
        P.fixed(lon, lat, s, **COUNTRY)
    P.fixed(51.75, 27.05, "Persian Gulf", rotation=-33, **SEA)
    P.fixed(HORMUZ[0] + 0.08, HORMUZ[3] - 0.08, "Fig. 3", ha="left", va="top", fontsize=6.5,
            style="italic")
    scalebar(ax, 56.9, 30.25, 100, placer=P)

    P.place(56.45, 26.55, "Strait of Hormuz", fontsize=7.1, fontweight="semibold",
            dists=(7, 10, 14, 19, 25), leader_from=99, prefer=("NE", "E", "N", "NW", "W", "SE", "S"))
    labels = [("Basra terminals", 48.81, 29.78), ("Abadan", 48.28, 30.34),
              ("Mina al-Ahmadi", 48.15, 29.07), ("Ali Al Salem (Kuwait)", 47.52, 29.35),
              ("Kharg Island", 50.31, 29.24), ("South Pars", 52.61, 27.48),
              ("Ras Tanura", 50.18, 26.70), ("Abqaiq", 49.68, 25.93), ("Khurais", 48.19, 25.07),
              ("Manama (Bahrain)", 50.61, 26.21), ("Ras Laffan (Qatar)", 51.56, 25.91),
              ("Al Udeid", 51.31, 25.12), ("Mesaieed", 51.56, 24.99), ("Das Island", 52.87, 25.15),
              ("Ruwais", 52.73, 24.11), ("Abu Dhabi", 54.50, 24.35), ("Dubai", 55.36, 25.25),
              ("Fujairah", 56.36, 25.17), ("Bandar Abbas", 56.21, 27.14), ("Jask", 57.77, 25.64),
              ("Sohar", 56.62, 24.49), ("Mina al-Fahal", 58.52, 23.63)]
    for s, lon, lat in labels:
        P.place(lon, lat, s, fontsize=6.4)

    h_route = Line2D([], [], color="#34495E", lw=1.0, linestyle=(0, (1.2, 1.6)),
                     label="Main shipping route (schematic)")
    fig.legend(handles=legend_handles() + [H_STAR, H_PIPE, h_route], loc="lower center",
               bbox_to_anchor=(0.53, 0.004), ncol=3, columnspacing=1.5, handletextpad=0.5,
               labelspacing=0.5, fontsize=6.9)
    print("placer failures:", P.failed)
    audit(fig, "fig2", symbols=[b for b, t in P.boxes if t == "symbol"])
    save(fig, "fig2_gulf")
    plt.close(fig)


# ============================================================== Figure 3
def fig3():
    fig = plt.figure(figsize=(TEXTWIDTH, 5.95))
    ax = fig.add_axes([0.085, 0.158, 0.86, 0.832], projection=ccrs.Mercator())
    basemap(ax, HORMUZ, minor_islands=True, xticks=np.arange(55, 59, 1), yticks=np.arange(24.5, 28, 0.5))

    route = np.array(CFG["route_schematic"])
    ax.plot(route[:, 0], route[:, 1], transform=PC, color="#34495E", lw=1.2,
            linestyle=(0, (5, 2.5)), zorder=3)
    styles = {"gate": dict(edgecolor=C["gate"], facecolor=C["gate"] + "20", lw=1.3, ls="-"),
              "queue_oman": dict(edgecolor=C["queue"], facecolor=C["queue"] + "14", lw=1.0, ls=(0, (3, 2))),
              "queue_gulf": dict(edgecolor=C["queue"], facecolor=C["queue"] + "14", lw=1.0, ls=(0, (3, 2)))}
    for key, st in styles.items():
        a = CFG["aoi"][key]
        ax.add_patch(Rectangle((a["lon_min"], a["lat_min"]), a["lon_max"] - a["lon_min"],
                               a["lat_max"] - a["lat_min"], transform=PC, zorder=2.5,
                               edgecolor=st["edgecolor"], facecolor=st["facecolor"],
                               linewidth=st["lw"], linestyle=st["ls"]))
    draw_infra(ax, HORMUZ, base=16, gain=20)
    draw_strikes(ax, HORMUZ, s=30)

    P = Placer(ax)
    for _, r in inside(INFRA, HORMUZ).iterrows():
        P.add_marker(r["lon"], r["lat"], (16 + 20 * np.sqrt(r["capacity_mboed"])) * 1.25)
    for _, r in inside(STRIKES, HORMUZ).iterrows():
        P.add_marker(r["lon"], r["lat"], 30 * 1.7)
    P.add_line(route[:, 0], route[:, 1], width_pt=1.2)
    for key in styles:
        a = CFG["aoi"][key]
        P.add_rect_outline(a["lon_min"], a["lon_max"], a["lat_min"], a["lat_max"])

    g, qo, qg = CFG["aoi"]["gate"], CFG["aoi"]["queue_oman"], CFG["aoi"]["queue_gulf"]
    P.fixed((g["lon_min"] + g["lon_max"]) / 2, g["lat_max"] - 0.075, "Strait of Hormuz",
            fontsize=7.4, fontweight="semibold", halo=True)
    P.fixed(g["lon_max"] + 0.06, g["lat_min"] + 0.09, "Radar transit gate", ha="left",
            fontsize=6.9, color="#8C1D18", fontweight="semibold", halo=True)
    P.fixed(qo["lon_min"] + 0.06, qo["lat_min"] + 0.09, "Queue zone, Gulf of Oman", ha="left",
            fontsize=6.6, color="#4A2A8F", halo=True)
    P.fixed(qg["lon_min"] + 0.06, qg["lat_min"] + 0.09, "Queue zone, inside the Gulf", ha="left",
            fontsize=6.6, color="#4A2A8F", halo=True)
    for s, lon, lat in [("IRAN", 57.7, 27.35), ("OMAN", 57.05, 24.45),
                        ("UNITED ARAB\nEMIRATES", 55.3, 24.62), ("MUSANDAM\n(OMAN)", 56.21, 25.99)]:
        P.fixed(lon, lat, s, linespacing=1.1, **COUNTRY)
    P.fixed(54.65, 26.3, "Persian Gulf", fontsize=7.2, color=C["seatext"], style="italic")
    P.fixed(58.1, 24.62, "Gulf of Oman", fontsize=7.2, color=C["seatext"], style="italic")
    P.fixed(55.85, 26.80, "Qeshm", fontsize=6.6, halo=True, check=False)
    scalebar(ax, 54.4, 27.42, 50, placer=P)

    for s, lon, lat in [("Bandar Abbas", 56.27, 27.19), ("Larak", 56.36, 26.86),
                        ("Hormuz Island", 56.46, 27.06), ("Khasab", 56.25, 26.18),
                        ("Dubai", 55.30, 25.27), ("Fujairah", 56.36, 25.17), ("Sohar", 56.62, 24.49),
                        ("Jask", 57.77, 25.64), ("Bandar Lengeh", 54.88, 26.56),
                        ("Abu Musa", 55.03, 25.87), ("Greater Tunb", 55.30, 26.26),
                        ("Tankers attacked, June 2019", 57.40, 25.45), ("Musaffah", 54.50, 24.35)]:
        P.place(lon, lat, s, fontsize=6.6)

    handles = [Patch(facecolor=C["gate"] + "20", edgecolor=C["gate"], lw=1.3, label="Radar transit gate (flow)"),
               Patch(facecolor=C["queue"] + "14", edgecolor=C["queue"], lw=1.0, linestyle=(0, (3, 2)),
                     label="Queue zones (waiting stock)"),
               Line2D([], [], color="#34495E", lw=1.2, linestyle=(0, (5, 2.5)), label="Main route (schematic)"),
               H_DEEP] + legend_handles()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.515, 0.004), ncol=3,
               columnspacing=1.5, handletextpad=0.5, labelspacing=0.5, fontsize=6.9)
    print("placer failures:", P.failed)
    audit(fig, "fig3", symbols=[b for b, t in P.boxes if t == "symbol"])
    save(fig, "fig3_hormuz")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    for k in which:
        {"1": fig1, "2": fig2, "3": fig3}[k]()
