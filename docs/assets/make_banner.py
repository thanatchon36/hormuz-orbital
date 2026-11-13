"""Dark README banner: the Strait of Hormuz at night. Run from repo root after fetch_basemap.py."""
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
NE = REPO / "data" / "naturalearth"
CFG = yaml.safe_load(open(REPO / "config.yaml"))
EXT = [48.6, 60.2, 23.6, 27.9]
PC = ccrs.PlateCarree()
BG, LAND, COAST, DEEP, AMBER, PALE = "#0b1020", "#18203a", "#4a5680", "#080c18", "#f5a623", "#ffe8a3"


def layer(name):
    g = gpd.read_file(NE / f"{name}.geojson")
    return g.cx[EXT[0] - 1:EXT[1] + 1, EXT[2] - 1:EXT[3] + 1]


fig = plt.figure(figsize=(16, 5), dpi=150, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.Mercator())
ax.set_extent(EXT, crs=PC)
ax.set_facecolor(BG); ax.spines["geo"].set_visible(False); ax.set_adjustable("datalim")
ax.add_geometries(layer("ne_10m_bathymetry_K_200").geometry, PC, facecolor=DEEP, edgecolor="none")
ax.add_geometries(layer("ne_10m_land").geometry, PC, facecolor=LAND, edgecolor=COAST, linewidth=0.6)
ax.add_geometries(layer("ne_10m_minor_islands").geometry, PC, facecolor=LAND, edgecolor=COAST, linewidth=0.5)
ax.add_geometries(layer("ne_10m_admin_0_boundary_lines_land").geometry, PC, facecolor="none",
                  edgecolor="#2a3150", linewidth=0.6, linestyle=(0, (3, 2)))

# Route as a glowing amber line, with a scatter of "vessels" along it (illustrative only).
r = np.array(CFG["route_schematic"])
for lw, a in [(14, 0.05), (8, 0.10), (4, 0.25), (1.6, 0.95)]:
    ax.plot(r[:, 0], r[:, 1], color=AMBER, lw=lw, alpha=a, transform=PC, solid_capstyle="round")
rng = np.random.default_rng(7)
t = np.linspace(0, 1, 400)
seg = np.r_[0, np.cumsum(np.hypot(*np.diff(r, axis=0).T))]
s = rng.uniform(0, seg[-1], 160)
x, y = np.interp(s, seg, r[:, 0]), np.interp(s, seg, r[:, 1])
x += rng.normal(0, 0.06, x.size); y += rng.normal(0, 0.05, y.size)
ax.scatter(x, y, s=rng.uniform(2, 14, x.size), color=PALE, alpha=0.8, linewidths=0, transform=PC, zorder=5)
ax.scatter(x, y, s=60, color=AMBER, alpha=0.06, linewidths=0, transform=PC, zorder=4)

g = CFG["aoi"]["gate"] if "aoi" in CFG else None
for k, v in CFG.items():
    if isinstance(v, dict) and "gate" in v and isinstance(v["gate"], dict):
        g = v["gate"]
if g:
    ax.add_patch(Rectangle((g["lon_min"], g["lat_min"]), g["lon_max"] - g["lon_min"], g["lat_max"] - g["lat_min"],
                           transform=PC, facecolor="none", edgecolor=AMBER, lw=1.4, ls=(0, (4, 2)), zorder=6))

fig.text(0.035, 0.80, "hormuz-orbital", color=PALE, fontsize=44, fontweight="bold", family="DejaVu Sans")
fig.text(0.037, 0.70, "Satellite-observed shipping, missile strikes and energy prices",
         color="#c9cfe0", fontsize=16)
fig.text(0.037, 0.08, "Strait of Hormuz · schematic route and transit gate · not for navigation",
         color="#6b7596", fontsize=10)
out = REPO / "docs" / "assets" / "banner.png"
fig.savefig(out, facecolor=BG)
print("saved", out)
