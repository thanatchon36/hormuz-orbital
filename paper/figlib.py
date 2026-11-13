"""
figlib.py -- house style and helpers for the paper's figures.

* style()            Inter typeface, hairline axes, vector PDF output
* basemap()          Natural Earth land/sea clipped to the extent, rasterised
                     underneath so that labels and symbols stay vector
* Placer             greedy, collision-free label placement with leader lines
* audit()            reports every pair of overlapping text boxes and every
                     label that overlaps a symbol or leaves the canvas
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
from shapely.geometry import box as sbox

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
NE = REPO / "data" / "naturalearth"
REF = REPO / "data" / "reference"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
PC = ccrs.PlateCarree()

TEXTWIDTH = 6.5          # inches, arxiv.sty

C = dict(
    sea="#EEF3F6", deep="#DDE8EF", land="#F8F6F1", coast="#8B959C", border="#BEB8AD",
    ink="#1F2933", muted="#65717B", grid="#CBD3D8", rule="#DDE2E6",
    ep1="#2F6DB5", ep2="#E0A019", ep3="#1E9C89", ep4="#CF4257",
    infra="#2B2D42", navy="#0B3C5D", orange="#D55E00", gold="#F4C430",
    gate="#B3261E", queue="#5E35B1", country="#9C9588", seatext="#5A86A5",
)
EPISODES = [("2019-2022 incidents", "2019–2022 incidents", C["ep1"]),
            ("2024 exchanges", "2024 exchanges", C["ep2"]),
            ("June 2025 war", "June 2025 war", C["ep3"]),
            ("2026 war", "2026 war (unverified)", C["ep4"])]
INFRA_MARKER = {"oil_terminal": "s", "oil_processing": "^", "oil_field": "^",
                "refinery": "^", "lng": "D", "gas_processing": "D"}
INFRA_LABEL = [("s", "Oil export terminal"), ("^", "Refinery or processing plant"),
               ("D", "LNG or gas processing")]
HALO = [pe.withStroke(linewidth=1.9, foreground="white", alpha=0.9)]


FONT_DIR = Path(__file__).resolve().parent / "fonts"


def style():
    from matplotlib import font_manager as fm
    files = sorted(FONT_DIR.glob("InterPaper-*.ttf"))
    for f in files:
        fm.fontManager.addfont(str(f))
    # Inter converted to TrueType (run paper/fonts/make_fonts.py) embeds most
    # cleanly; otherwise fall back to an installed Inter, then DejaVu Sans.
    names = {f.name for f in fm.fontManager.ttflist}
    family = "Inter Paper" if files else ("Inter" if "Inter" in names else "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": family, "font.size": 7.5,
        "axes.linewidth": 0.6, "axes.edgecolor": C["ink"], "axes.labelsize": 7.5,
        "axes.labelcolor": C["ink"], "axes.titlesize": 8, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.color": C["ink"], "ytick.color": C["ink"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": 7, "legend.frameon": False, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400, "figure.dpi": 200,
        "savefig.facecolor": "white", "mathtext.fontset": "custom", "mathtext.rm": family,
        "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
    })


# ------------------------------------------------------------------ basemap
_raw, _clip = {}, {}


def layer(name, extent, pad=0.4):
    key = (name, tuple(extent))
    if key not in _clip:
        if name not in _raw:
            _raw[name] = gpd.read_file(NE / f"{name}.geojson")
        x0, x1, y0, y1 = extent
        _clip[key] = gpd.clip(_raw[name], sbox(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return _clip[key]


def basemap(ax, extent, *, minor_islands=False, xticks=None, yticks=None, lakes=True,
            borders=True, coast_lw=0.45, ticklabels=True):
    """Land, sea, deep water, borders and a labelled graticule on a Mercator axes."""
    ax.set_extent(extent, crs=PC)
    ax.set_facecolor(C["sea"])
    deep = layer("ne_10m_bathymetry_K_200", extent)
    if len(deep):
        ax.add_geometries(deep.geometry, PC, facecolor=C["deep"], edgecolor="none", zorder=0.5)
    ax.add_geometries(layer("ne_10m_land", extent).geometry, PC, facecolor=C["land"],
                      edgecolor=C["coast"], linewidth=coast_lw, zorder=1.0)
    if minor_islands:
        mi = layer("ne_10m_minor_islands", extent, pad=0.1)
        if len(mi):
            ax.add_geometries(mi.geometry, PC, facecolor=C["land"], edgecolor=C["coast"],
                              linewidth=coast_lw * 0.9, zorder=1.0)
    if lakes:
        lk = layer("ne_10m_lakes", extent)
        if len(lk):
            ax.add_geometries(lk.geometry, PC, facecolor=C["sea"], edgecolor=C["coast"],
                              linewidth=0.3, zorder=1.1)
    if borders:
        bl = layer("ne_10m_admin_0_boundary_lines_land", extent)
        if len(bl):
            ax.add_geometries(bl.geometry, PC, facecolor="none", edgecolor=C["border"],
                              linewidth=0.5, zorder=1.2)
    ax.set_rasterization_zorder(1.3)          # raster below, vector above
    if xticks is not None:
        ax.gridlines(xlocs=xticks, ylocs=yticks, linewidth=0.3, color=C["grid"], alpha=0.75,
                     linestyle=(0, (2, 3)), zorder=1.35)
        if ticklabels:
            ax.set_xticks(xticks, crs=PC)
            ax.set_yticks(yticks, crs=PC)
            ax.xaxis.set_major_formatter(LongitudeFormatter(number_format="g"))
            ax.yaxis.set_major_formatter(LatitudeFormatter(number_format="g"))
            ax.tick_params(length=2.5, width=0.5, color=C["muted"], labelsize=6.5,
                           labelcolor=C["muted"], pad=2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor("#6F7B84")
        s.set_linewidth(0.6)
    return ax


def scalebar(ax, lon, lat, km, placer=None):
    dlon = km / (111.32 * np.cos(np.radians(lat)))
    y0, y1 = ax.get_extent(PC)[2:]
    tick = 0.012 * (y1 - y0)
    ax.plot([lon, lon + dlon], [lat, lat], transform=PC, color=C["ink"], lw=1.4,
            solid_capstyle="butt", zorder=7)
    for x in (lon, lon + dlon):
        ax.plot([x, x], [lat, lat + tick], transform=PC, color=C["ink"], lw=0.8, zorder=7)
    t = ax.text(lon + dlon / 2, lat + 1.7 * tick, f"{km:g} km", transform=PC, ha="center",
                va="bottom", fontsize=6.5, color=C["ink"], zorder=7)
    if placer is not None:
        placer.add_text(t)
        placer.add_line([lon, lon + dlon], [lat, lat], width_pt=2)
    return t


# ------------------------------------------------------------- label placer
class Placer:
    """Greedy label placement in display space.

    Register everything that must stay clear (symbols, fixed text, lines),
    then call place() for each label: it tries eight compass positions at
    increasing distances and keeps the first that touches nothing, adding a
    leader line when the label had to move away from its point.
    """

    DIRS = {"E": (1, 0, "left", "center"), "W": (-1, 0, "right", "center"),
            "N": (0, 1, "center", "bottom"), "S": (0, -1, "center", "top"),
            "NE": (0.74, 0.74, "left", "bottom"), "NW": (-0.74, 0.74, "right", "bottom"),
            "SE": (0.74, -0.74, "left", "top"), "SW": (-0.74, -0.74, "right", "top")}

    def __init__(self, ax, margin_pt=1.3):
        self.ax, self.fig = ax, ax.figure
        self.fig.canvas.draw()
        self.r = self.fig.canvas.get_renderer()
        self.k = self.fig.dpi / 72.0
        self.m = margin_pt * self.k
        self.axbox = ax.get_window_extent(self.r)
        self.boxes: list[tuple[Bbox, str]] = []
        self.failed: list[str] = []
        self.tr = PC._as_mpl_transform(ax)

    # ---- registration
    def disp(self, lon, lat):
        x, y = self.ax.projection.transform_point(lon, lat, PC)
        return self.ax.transData.transform((x, y))

    def add_box(self, bb, tag=""):
        self.boxes.append((bb, tag))

    def add_marker(self, lon, lat, s_pt2, tag="symbol"):
        d = np.sqrt(s_pt2) * self.k * 1.05
        x, y = self.disp(lon, lat)
        self.add_box(Bbox.from_extents(x - d / 2, y - d / 2, x + d / 2, y + d / 2), tag)

    def add_text(self, t, tag=None):
        self.add_box(Text.get_window_extent(t, self.r), tag or t.get_text())

    def add_line(self, lons, lats, width_pt=1.2, step_px=5, tag="line"):
        pts = np.array([self.disp(a, b) for a, b in zip(lons, lats)])
        w = width_pt * self.k
        for p, q in zip(pts[:-1], pts[1:]):
            n = max(int(np.hypot(*(q - p)) / step_px), 1)
            for f in np.linspace(0, 1, n + 1):
                x, y = p + f * (q - p)
                self.add_box(Bbox.from_extents(x - w, y - w, x + w, y + w), tag)

    def add_rect_outline(self, lon0, lon1, lat0, lat1, width_pt=1.2, tag="frame"):
        self.add_line([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                      width_pt=width_pt, tag=tag)

    # ---- tests
    def free(self, bb, inside=True):
        b = Bbox.from_extents(bb.x0 - self.m, bb.y0 - self.m, bb.x1 + self.m, bb.y1 + self.m)
        a = self.axbox
        if inside and not (b.x0 >= a.x0 and b.x1 <= a.x1 and b.y0 >= a.y0 and b.y1 <= a.y1):
            return False
        for o, _ in self.boxes:
            if b.x0 < o.x1 and o.x0 < b.x1 and b.y0 < o.y1 and o.y0 < b.y1:
                return False
        return True

    # ---- fixed text (must already be clear; reported if not)
    def fixed(self, lon, lat, s, *, halo=False, check=True, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center")
        kw.setdefault("fontsize", 6.5); kw.setdefault("color", C["ink"]); kw.setdefault("zorder", 6)
        t = self.ax.text(lon, lat, s, transform=PC, **kw)
        if halo:
            t.set_path_effects(HALO)
        bb = Text.get_window_extent(t, self.r)
        if check and not self.free(bb):
            self.failed.append(f"fixed:{s!r}")
        self.add_box(bb, s)
        return t

    # ---- automatic placement
    def place(self, lon, lat, s, *, prefer=("E", "W", "NE", "SE", "NW", "SW", "N", "S"),
              dists=(4.5, 7.5, 11, 15, 20, 26, 33), leader_from=10.0, halo=True, **kw):
        kw.setdefault("fontsize", 6.3); kw.setdefault("color", C["ink"]); kw.setdefault("zorder", 8)
        for d in dists:
            for key in prefer:
                ux, uy, ha, va = self.DIRS[key]
                t = self.ax.annotate(s, xy=(lon, lat), xycoords=self.tr, xytext=(ux * d, uy * d),
                                     textcoords="offset points", ha=ha, va=va,
                                     annotation_clip=False, **kw)
                bb = t.get_window_extent(self.r)      # Annotation: positions updated first
                if self.free(bb):
                    t.remove()
                    arrow = None
                    if d >= leader_from:
                        arrow = dict(arrowstyle="-", lw=0.45, color=C["muted"], shrinkA=1.0,
                                     shrinkB=2.6, connectionstyle="arc3,rad=0")
                    t = self.ax.annotate(s, xy=(lon, lat), xycoords=self.tr,
                                         xytext=(ux * d, uy * d), textcoords="offset points",
                                         ha=ha, va=va, annotation_clip=False,
                                         arrowprops=arrow, **kw)
                    if halo:
                        t.set_path_effects(HALO)
                    self.add_box(bb, s)
                    return t
                t.remove()
        self.failed.append(f"auto:{s!r}")
        return None


# ------------------------------------------------------------------- audit
def audit(fig, name="figure", pad_px=0.5, symbols=None, verbose=True):
    """Check a finished figure for overlapping or clipped text.

    Returns (text-text overlaps, text-symbol overlaps, out-of-canvas labels).
    `symbols` is an optional list of display-space Bbox objects.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        ax = t.axes
        if ax is not None and not ax.get_visible():
            continue
        if isinstance(t, Annotation):
            t.update_positions(r)
        bb = Text.get_window_extent(t, r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((s.replace("\n", " / "), bb))
    tt = []
    for i in range(len(items)):
        a = items[i][1]
        for j in range(i + 1, len(items)):
            b = items[j][1]
            if (a.x0 < b.x1 - pad_px and b.x0 < a.x1 - pad_px
                    and a.y0 < b.y1 - pad_px and b.y0 < a.y1 - pad_px):
                tt.append((items[i][0], items[j][0]))
    ts = []
    for s, a in items:
        for b in symbols or []:
            if (a.x0 < b.x1 - pad_px and b.x0 < a.x1 - pad_px
                    and a.y0 < b.y1 - pad_px and b.y0 < a.y1 - pad_px):
                ts.append(s)
                break
    w, h = fig.canvas.get_width_height()
    out = [s for s, bb in items if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > w + 1 or bb.y1 > h + 1]
    if verbose:
        print(f"[audit] {name}: {len(items)} text items | text-text overlaps: {len(tt)} | "
              f"text-symbol overlaps: {len(ts)} | off-canvas: {len(out)}")
        for a, b in tt:
            print("    overlap:", repr(a), "<->", repr(b))
        for s in ts:
            print("    on symbol:", repr(s))
        for s in out:
            print("    off-canvas:", repr(s))
    return tt, ts, out


def save(fig, stem, **kw):
    """Vector PDF for the paper and a PNG preview for inspection."""
    fig.savefig(OUT / f"{stem}.pdf", **kw)
    fig.savefig(OUT / f"{stem}.png", dpi=220, **kw)
    return OUT / f"{stem}.pdf"
