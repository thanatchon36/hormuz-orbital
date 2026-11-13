"""
s1_gee_pipeline.py -- Sentinel-1 vessel detection in Google Earth Engine.

STATUS: NOT RUN. The build environment had no Earth Engine account and no
route to Google. The code follows the Earth Engine Python API as the author
knows it; expect to fix small things on first run. The statistical logic is
the same CA-CFAR that is tested in s1_cfar.py, without the hysteresis step.

Two routes, use both:

  A. Server-side detection (cheap, whole archive)
       export_detections("gate", "2023-01-01", "2026-11-01")
     writes one GeoJSON of detection polygons and one CSV of scene coverage
     to Google Drive. flow_index.detections_to_counts() turns them into
     per-scene vessel counts.

  B. Image chips for the local detector (validation subset)
       export_chips("gate", "2026-02-15", "2026-03-15")
     writes VH/VV GeoTIFFs; run s1_cfar.detect_vessels() on them and compare
     with route A.

Setup:  pip install earthengine-api ;  earthengine authenticate
        ee_init("your-cloud-project-id")
"""
from __future__ import annotations

from pathlib import Path

import yaml
from scipy.special import gammainccinv

try:
    import ee
except ImportError:                      # lets the module be imported without ee
    ee = None

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
S = CFG["sar"]
HANSEN = "UMD/hansen/global_forest_change_2023_v1_11"   # datamask: 1 = land


def ee_init(project: str):
    ee.Initialize(project=project)


def box(a: dict):
    """Planar lon/lat rectangle from a config AOI."""
    return ee.Geometry.Rectangle(
        [a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"]], None, False)


def s1_collection(aoi, start: str, end: str):
    """Sentinel-1 IW GRD scenes with both VV and VH over the AOI."""
    return (ee.ImageCollection(S["collection"])
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", S["instrument_mode"]))
            .filter(ee.Filter.eq("resolution_meters", S["pixel_m"]))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))


def sea_mask(aoi, reference_year: int):
    """1 = open sea that may contain a vessel.

    Removes (i) land plus a coastal buffer and (ii) fixed bright structures
    (platforms, islets, breakwaters), found as pixels whose median VH over a
    reference year exceeds `static_db`. Use the year BEFORE the analysis
    window so that the mask does not learn from the period under study.
    """
    land = ee.Image(HANSEN).select("datamask").eq(1).unmask(0)
    near_land = land.focalMax(radius=S["coast_buffer_m"], kernelType="circle",
                              units="meters")
    median_vh = (s1_collection(aoi, f"{reference_year}-01-01", f"{reference_year}-12-31")
                 .select(S["band"]).median())
    static = (median_vh.gt(S["static_db"]).unmask(0)
              .focalMax(radius=100, kernelType="circle", units="meters"))
    return near_land.Or(static).Not().rename("sea")


def _box_sum(image, radius_px: int):
    """Sum over a (2r+1)^2 window. Band order is preserved."""
    names = image.bandNames()
    return (image.reduceNeighborhood(
                reducer=ee.Reducer.sum(),
                kernel=ee.Kernel.square(radius_px, "pixels", False),
                optimization="boxcar")
            .rename(names))


def detect_scene(img, aoi, sea):
    """CA-CFAR on one scene. Returns a FeatureCollection of detection polygons."""
    band_db = img.select(S["band"])
    angle = img.select("angle")
    # trim swath edges, where border noise breaks the clutter model
    usable = angle.gt(31).And(angle.lt(45)).And(band_db.gt(-45))
    valid = sea.And(usable).And(band_db.mask()).unmask(0)
    lin = ee.Image(10).pow(band_db.divide(10))
    x = lin.updateMask(valid).unmask(0)

    stack = x.addBands(valid.toFloat()).rename(["s1", "n"])
    r_bg = int(round(S["background_m"] / S["pixel_m"]))
    r_g = int(round(S["guard_m"] / S["pixel_m"]))
    ring = _box_sum(stack, r_bg).subtract(_box_sum(stack, r_g))
    n = ring.select("n")
    mu = ring.select("s1").divide(n)

    t = float(gammainccinv(S["enl"], S["pfa"]) / S["enl"])       # Gamma CA-CFAR
    ring_area = (2 * r_bg + 1) ** 2 - (2 * r_g + 1) ** 2
    det = (x.gt(mu.multiply(t)).And(valid).And(n.gte(0.3 * ring_area)))
    # close one-pixel gaps so that a hull is one object
    det = (det.focalMax(radius=1, kernelType="square", units="pixels")
              .focalMin(radius=1, kernelType="square", units="pixels"))

    vectors = (det.selfMask().toByte().rename("det")
               .addBands(band_db.rename("mean_db"))
               .reduceToVectors(geometry=aoi, scale=S["pixel_m"],
                                geometryType="polygon", eightConnected=True,
                                labelProperty="det", reducer=ee.Reducer.mean(),
                                maxPixels=1e10, tileScale=4))
    props = {"scene_id": img.get("system:index"),
             "time_start": img.get("system:time_start"),
             "orbit_pass": img.get("orbitProperties_pass"),
             "rel_orbit": img.get("relativeOrbitNumber_start"),
             "platform": img.get("platform_number")}
    return vectors.map(lambda f: f.set(props))


def scene_coverage(img, aoi):
    """Share of the AOI inside the scene footprint (features without geometry)."""
    cov = img.geometry().intersection(aoi, 10).area(10).divide(aoi.area(10))
    return ee.Feature(None, {"scene_id": img.get("system:index"),
                             "time_start": img.get("system:time_start"),
                             "orbit_pass": img.get("orbitProperties_pass"),
                             "rel_orbit": img.get("relativeOrbitNumber_start"),
                             "platform": img.get("platform_number"),
                             "coverage": cov})


def export_detections(aoi_name: str, start: str, end: str, folder: str = "hormuz_orbital"):
    """Route A. Start two Drive export tasks: detections (GeoJSON) and scene
    coverage (CSV). Export year by year to keep each task small."""
    aoi = box(CFG["aoi"][aoi_name])
    sea = sea_mask(aoi, int(start[:4]) - 1)
    col = s1_collection(aoi, start, end)
    dets = ee.FeatureCollection(col.map(lambda im: detect_scene(im, aoi, sea))).flatten()
    cov = ee.FeatureCollection(col.map(lambda im: scene_coverage(im, aoi)))
    tag = f"{aoi_name}_{start}_{end}".replace("-", "")
    tasks = [
        ee.batch.Export.table.toDrive(collection=dets, description=f"det_{tag}",
                                      folder=folder, fileFormat="GeoJSON"),
        ee.batch.Export.table.toDrive(collection=cov, description=f"cov_{tag}",
                                      folder=folder, fileFormat="CSV",
                                      selectors=["scene_id", "time_start", "orbit_pass",
                                                 "rel_orbit", "platform", "coverage"]),
    ]
    for t in tasks:
        t.start()
    return tasks


def export_chips(aoi_name: str, start: str, end: str, folder: str = "hormuz_orbital",
                 scale: float | None = None, crs: str = "EPSG:32640", max_scenes: int | None = None):
    """Route B. Export VH/VV chips (dB, float32) and the sea mask as GeoTIFFs.
    EPSG:32640 is UTM zone 40N, which covers the Strait."""
    aoi = box(CFG["aoi"][aoi_name])
    scale = scale or S["pixel_m"]
    col = s1_collection(aoi, start, end)
    ids = col.aggregate_array("system:index").getInfo()
    tasks = [ee.batch.Export.image.toDrive(
        image=sea_mask(aoi, int(start[:4]) - 1).toByte(),
        description=f"seamask_{aoi_name}", folder=folder, region=aoi, scale=scale,
        crs=crs, maxPixels=1e10, fileFormat="GeoTIFF")]
    for sid in ids[:max_scenes]:
        img = ee.Image(f"{S['collection']}/{sid}").select([S["band"], S["confirm_band"]])
        tasks.append(ee.batch.Export.image.toDrive(
            image=img.toFloat(), description=f"chip_{aoi_name}_{sid}", folder=folder,
            region=aoi, scale=scale, crs=crs, maxPixels=1e10, fileFormat="GeoTIFF"))
    for t in tasks:
        t.start()
    return tasks


def revisit_table(aoi_name: str, start: str, end: str):
    """Acquisition dates, platform and pass direction over an AOI, as a list
    of dicts. Run this first: it tells you the true revisit cadence."""
    aoi = box(CFG["aoi"][aoi_name])
    fc = ee.FeatureCollection(s1_collection(aoi, start, end)
                              .map(lambda im: scene_coverage(im, aoi)))
    return [f["properties"] for f in fc.getInfo()["features"]]
