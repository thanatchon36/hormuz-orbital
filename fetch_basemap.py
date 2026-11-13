"""Download the Natural Earth 1:10m layers used by the maps (about 40 MB)
into data/naturalearth. Run once.  python fetch_basemap.py"""
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
LAYERS = ["ne_10m_land", "ne_10m_minor_islands", "ne_10m_lakes",
          "ne_10m_admin_0_boundary_lines_land", "ne_10m_bathymetry_K_200"]

out = Path(__file__).resolve().parent / "data" / "naturalearth"
out.mkdir(parents=True, exist_ok=True)
for name in LAYERS:
    path = out / f"{name}.geojson"
    if path.exists():
        print("have", path.name)
        continue
    r = requests.get(BASE + f"{name}.geojson", timeout=300)
    r.raise_for_status()
    path.write_bytes(r.content)
    print("saved", path.name, f"{len(r.content) / 1e6:.1f} MB")
