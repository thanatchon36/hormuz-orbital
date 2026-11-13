"""LinkedIn poster (1080x1350). Run from repo root after fetch_basemap.py."""
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

R = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(R / "config.yaml"))
BG, LAND, COAST, DEEP = "#0b1020", "#18203a", "#4a5680", "#080c18"
AMBER, PALE, INK2, MUTE = "#f5a623", "#ffe8a3", "#c9cfe0", "#6b7596"
TEAL = "#5fd3c6"
plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor=BG)

# ---------- map (top) ----------
EXT = [53.2, 59.6, 24.3, 27.3]
ax = fig.add_axes([0, 0.515, 1, 0.255], projection=ccrs.Mercator())
ax.set_extent(EXT, crs=ccrs.PlateCarree()); ax.set_adjustable("datalim")
ax.set_facecolor(BG); ax.spines["geo"].set_visible(False)
PC = ccrs.PlateCarree()
def layer(n):
    g = gpd.read_file(R / "data/naturalearth" / f"{n}.geojson")
    return g.cx[EXT[0]-2:EXT[1]+2, EXT[2]-2:EXT[3]+2]
ax.add_geometries(layer("ne_10m_bathymetry_K_200").geometry, PC, facecolor=DEEP, edgecolor="none")
ax.add_geometries(layer("ne_10m_land").geometry, PC, facecolor=LAND, edgecolor=COAST, lw=0.6)
ax.add_geometries(layer("ne_10m_minor_islands").geometry, PC, facecolor=LAND, edgecolor=COAST, lw=0.5)
r = np.array(CFG["route_schematic"])
for lw, a in [(16, .05), (9, .10), (4.5, .25), (1.8, .95)]:
    ax.plot(r[:, 0], r[:, 1], color=AMBER, lw=lw, alpha=a, transform=PC, solid_capstyle="round")
rng = np.random.default_rng(7)
seg = np.r_[0, np.cumsum(np.hypot(*np.diff(r, axis=0).T))]
s = rng.uniform(0, seg[-1], 140)
x = np.interp(s, seg, r[:, 0]) + rng.normal(0, .05, s.size)
y = np.interp(s, seg, r[:, 1]) + rng.normal(0, .04, s.size)
ax.scatter(x, y, s=rng.uniform(2, 12, s.size), color=PALE, alpha=.85, lw=0, transform=PC, zorder=5)
g = CFG["aoi"]["gate"]
ax.add_patch(Rectangle((g["lon_min"], g["lat_min"]), g["lon_max"]-g["lon_min"], g["lat_max"]-g["lat_min"],
             transform=PC, fc="none", ec=AMBER, lw=1.3, ls=(0, (4, 2)), zorder=6))
ax.text(56.95, 26.85, "transit gate", color=AMBER, fontsize=8, transform=PC)
fig.text(.06, .968, "NEW PREPRINT · NOVEMBER 2026", color=AMBER, fontsize=9.5, fontweight="bold")
fig.text(.06, .945, "Orbital Evidence on", color=PALE, fontsize=27, fontweight="bold", va="top")
fig.text(.06, .893, "Chokepoint Risk", color=AMBER, fontsize=27, fontweight="bold", va="top")
fig.text(.06, .832, "Do missile strikes move oil prices by taking barrels off the water,\nor by raising a risk premium? Satellites can tell the two apart.",
         color=INK2, fontsize=10.5, va="top", linespacing=1.35)

# ---------- stats ----------
stats = [("−95%", "tanker transits\nthrough Hormuz", "47.6 → 2.4 per day"),
         ("+94%", "Brent in six weeks", r"\$71 → \$138 / bbl"),
         ("+52.8", "log-pt Brent move\nby day 20", "placebo p = 0.015")]
for i, (big, lab, sub) in enumerate(stats):
    x0 = .06 + i * .31
    fig.text(x0, .475, big, color=AMBER if i < 2 else TEAL, fontsize=30, fontweight="bold", va="top")
    fig.text(x0, .41, lab, color=PALE, fontsize=9.5, va="top", linespacing=1.25)
    fig.text(x0, .362, sub, color=MUTE, fontsize=8.5, va="top")
fig.add_artist(plt.Line2D([.06, .94], [.495, .495], color="#2a3150", lw=1))

# ---------- chart ----------
pw = pd.read_csv(R / "data/raw/portwatch_daily_2023_2026_mirror.csv", parse_dates=["date"])
pw = pw[pw.portname.str.contains("Hormuz")].set_index("date")["n_tanker"].rolling(7).mean()
br = pd.read_csv(R / "data/raw/brent-daily.csv", parse_dates=["Date"]).set_index("Date")["Price"]
lo = pd.Timestamp("2025-06-01"); pw, br = pw[pw.index >= lo], br[br.index >= lo]
c = fig.add_axes([.10, .085, .78, .22], facecolor=BG)
c.fill_between(pw.index, pw.values, color=AMBER, alpha=.18, lw=0)
c.plot(pw.index, pw.values, color=AMBER, lw=1.6)
c2 = c.twinx(); c2.plot(br.index, br.values, color=TEAL, lw=1.6)
for a in (c, c2):
    for sp in a.spines.values(): sp.set_visible(False)
    a.tick_params(colors=MUTE, labelsize=7.5, length=0)
c.grid(axis="y", color="#1c2340", lw=.8); c.set_ylim(0, None); c2.set_ylim(40, None)
c.axvline(pd.Timestamp("2026-02-28"), color=PALE, lw=.8, ls=(0, (3, 2)))
c.text(pd.Timestamp("2026-03-08"), 8, "28 Feb 2026:\ntransits collapse", color=PALE, fontsize=7.5, va="bottom")
import matplotlib.dates as md
c.xaxis.set_major_formatter(md.DateFormatter("%b %y")); c.xaxis.set_major_locator(md.MonthLocator(interval=3))
fig.text(.10, .325, "Tankers / day through Hormuz (7-day avg)", color=AMBER, fontsize=8.5)
fig.text(.88, .325, r"Brent, \$/bbl", color=TEAL, fontsize=8.5, ha="right")

fig.text(.06, .028, "github.com/thanatchon36/hormuz-orbital  ·  DOI 10.5281/zenodo.24184102",
         color=MUTE, fontsize=7.8)
fig.text(.94, .028, "Thanatchon Chongmankhong", color=INK2, fontsize=7.8, ha="right")
fig.text(.06, .008, "Data: IMF PortWatch (AIS), U.S. EIA. Sentinel-1 index in progress. Map schematic, not for navigation.",
         color="#4a5270", fontsize=6.3)
out = R / "docs/linkedin/poster.png"
fig.savefig(out, facecolor=BG); print("saved", out)
