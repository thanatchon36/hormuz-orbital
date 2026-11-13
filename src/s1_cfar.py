"""
s1_cfar.py -- constant-false-alarm-rate (CFAR) vessel detection for SAR
backscatter such as Sentinel-1 GRD.

Statistical model
-----------------
Multilooked SAR intensity over homogeneous sea is Gamma distributed with shape
L (the equivalent number of looks, ENL) and mean mu. A pixel is called a target
when

    x > t(L, PFA) * mu_hat ,

where mu_hat is the mean of a background ring that surrounds a guard window
centred on the pixel, and t solves P(X > t * mu) = PFA for X ~ Gamma(L, mu / L).
This is the cell-averaging CFAR (CA-CFAR). Three variants are implemented:

    method="gamma"           fixed ENL (default; correct for homogeneous sea)
    method="gamma_adaptive"  local ENL from the ring's mean and variance
                             (more conservative over textured sea)
    method="two_parameter"   x > mu_hat + k * sigma_hat (classic rule)

Land and fixed structures are excluded through a boolean sea mask. Detections
are censored from the background and the statistics recomputed (n_iter=2), so
that neighbouring ships in an anchorage do not hide one another.

Input must be LINEAR power (sigma0), not dB:  lin = 10 ** (dB / 10).

Tested on synthetic Gamma clutter in tests/test_cfar_synthetic.py. Thresholds
have NOT yet been tuned on real Sentinel-1 scenes of the Strait.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.special import gammainccinv

__all__ = [
    "gamma_threshold_multiplier", "ring_stats", "cfar_detect",
    "extract_objects", "detect_vessels", "synth_scene", "score_detections",
]


# --------------------------------------------------------------------- CFAR
def gamma_threshold_multiplier(enl, pfa: float):
    """t such that P(X > t * E[X]) = pfa for X ~ Gamma(shape=enl)."""
    enl = np.asarray(enl, dtype=float)
    return gammainccinv(enl, pfa) / enl


def _box_sum(a: np.ndarray, half: int) -> np.ndarray:
    size = 2 * half + 1
    return ndi.uniform_filter(a, size=size, mode="constant", cval=0.0) * (size * size)


def ring_stats(x: np.ndarray, valid: np.ndarray, guard: int, bg: int):
    """Mean, variance and pixel count in the ring (bg window minus guard window).

    Only pixels where `valid` is True contribute. Half-widths are in pixels.
    """
    v = valid.astype(np.float64)
    xv = np.where(valid, x, 0.0).astype(np.float64)
    n = np.rint(_box_sum(v, bg) - _box_sum(v, guard))
    s1 = _box_sum(xv, bg) - _box_sum(xv, guard)
    s2 = _box_sum(xv * xv, bg) - _box_sum(xv * xv, guard)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        var = s2 / n - mu * mu
    mu[n <= 0] = np.nan
    var = np.where(n > 1, np.maximum(var, 0.0), np.nan)
    return mu, var, n


def _threshold(mu, var, *, method, enl, pfa, k, grid=None):
    """Per-pixel detection threshold for one false-alarm rate."""
    if method == "gamma":
        return gamma_threshold_multiplier(enl, pfa) * mu
    if method == "gamma_adaptive":
        l_grid, t_grid = grid
        with np.errstate(invalid="ignore", divide="ignore"):
            l_hat = np.clip(mu * mu / var, l_grid[0], l_grid[-1])
        return np.interp(np.nan_to_num(l_hat, nan=l_grid[0]), l_grid, t_grid) * mu
    if method == "two_parameter":
        return mu + k * np.sqrt(var)
    raise ValueError(f"unknown method {method!r}")


def _hysteresis(seed, grow):
    """Keep connected components of `grow` that contain at least one seed pixel."""
    lab, n = ndi.label(grow, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return seed
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[seed])] = True
    keep[0] = False
    return keep[lab]


def cfar_detect(x, sea_mask=None, *, pixel_m=10.0, guard_m=250.0,
                background_m=600.0, enl=4.4, pfa=1e-7, pfa_grow=1e-3,
                method="gamma", k=5.0, k_grow=3.0, min_ring_frac=0.3, n_iter=2):
    """Pixel-level CFAR detection with hysteresis.

    Seeds are pixels above the strict threshold (`pfa`, or `k` for the
    two-parameter rule). Each seed grows into connected pixels above a relaxed
    threshold (`pfa_grow`, or `k_grow`), which recovers whole hulls without
    raising the object-level false-alarm rate. Set pfa_grow=None to disable.

    Returns (detection mask, background mean used for the final pass).
    """
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x) & (x > 0)
    if sea_mask is not None:
        valid &= np.asarray(sea_mask, dtype=bool)
    g = int(round(guard_m / pixel_m))
    b = int(round(background_m / pixel_m))
    if b <= g:
        raise ValueError("background window must be larger than guard window")
    ring_area = (2 * b + 1) ** 2 - (2 * g + 1) ** 2

    grid = grid_grow = None
    if method == "gamma_adaptive":
        l_grid = np.linspace(0.3, float(enl), 96)
        grid = (l_grid, gamma_threshold_multiplier(l_grid, pfa))
        if pfa_grow is not None:
            grid_grow = (l_grid, gamma_threshold_multiplier(l_grid, pfa_grow))
    grow_on = (k_grow is not None) if method == "two_parameter" else (pfa_grow is not None)

    bg_valid = valid.copy()
    det = np.zeros_like(valid)
    mu = np.full(x.shape, np.nan)
    for _ in range(max(1, n_iter)):
        mu, var, n = ring_stats(x, bg_valid, g, b)
        ok = valid & (n >= min_ring_frac * ring_area)
        thr = _threshold(mu, var, method=method, enl=enl, pfa=pfa, k=k, grid=grid)
        with np.errstate(invalid="ignore"):
            det = ok & (x > thr)
            if grow_on:
                thr_g = _threshold(mu, var, method=method, enl=enl, pfa=pfa_grow,
                                   k=k_grow, grid=grid_grow)
                det = _hysteresis(det, ok & (x > thr_g))
        # censor detections (slightly dilated) from the next background pass
        bg_valid = valid & ~ndi.binary_dilation(det, iterations=2)
    return det, mu


# ------------------------------------------------------------------ objects
def _pixel_to_xy(transform, row, col):
    """Affine (a, b, c, d, e, f): x = a*col + b*row + c ; y = d*col + e*row + f.
    Accepts a 6-tuple or a rasterio Affine. Pixel centres are used."""
    a, b, c, d, e, f = tuple(transform)[:6]
    return a * (col + 0.5) + b * (row + 0.5) + c, d * (col + 0.5) + e * (row + 0.5) + f


def extract_objects(det, x, mu, *, pixel_m=10.0, min_pixels=4,
                    min_length_m=60.0, max_length_m=480.0, transform=None):
    """Group detected pixels into objects and measure them.

    Length and width come from the second moments of the pixel coordinates,
    assuming a filled rectangle: side = sqrt(12 * variance + 1) pixels.
    SAR sidelobes and motion smear inflate these, so calibrate against AIS
    lengths before using size classes.
    """
    cols = ["row", "col", "area_px", "length_m", "width_m", "heading_deg",
            "peak", "tcr_db", "x", "y"]
    lab, n_obj = ndi.label(det, structure=np.ones((3, 3), dtype=int))
    if n_obj == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if area < min_pixels:
            continue
        rr, cc = np.nonzero(m)
        r0, c0 = rr.mean() + sl[0].start, cc.mean() + sl[1].start
        if area >= 2:
            cov = np.cov(np.vstack([cc, rr]).astype(float), bias=True)
            evals, evecs = np.linalg.eigh(cov)
            lam_minor, lam_major = max(evals[0], 0.0), max(evals[1], 0.0)
            vx, vy = evecs[:, 1]                       # major axis (col, row)
            heading = np.degrees(np.arctan2(vx, -vy)) % 180.0   # 0 = up/north
        else:
            lam_minor = lam_major = 0.0
            heading = np.nan
        length = np.sqrt(12.0 * lam_major + 1.0) * pixel_m
        width = np.sqrt(12.0 * lam_minor + 1.0) * pixel_m
        if not (min_length_m <= length <= max_length_m):
            continue
        xs, bs = x[sl][m], mu[sl][m]
        with np.errstate(invalid="ignore", divide="ignore"):
            tcr = 10.0 * np.log10(np.nanmean(xs) / np.nanmean(bs))
        px, py = (np.nan, np.nan)
        if transform is not None:
            px, py = _pixel_to_xy(transform, r0, c0)
        rows.append((r0, c0, area, length, width, heading, float(xs.max()), tcr, px, py))
    return pd.DataFrame(rows, columns=cols)


def detect_vessels(x, sea_mask=None, *, pixel_m=10.0, transform=None,
                   min_pixels=4, min_length_m=60.0, max_length_m=480.0, **cfar_kw):
    """Convenience wrapper: CFAR then object extraction. Returns a DataFrame."""
    det, mu = cfar_detect(x, sea_mask, pixel_m=pixel_m, **cfar_kw)
    return extract_objects(det, x, mu, pixel_m=pixel_m, min_pixels=min_pixels,
                           min_length_m=min_length_m, max_length_m=max_length_m,
                           transform=transform)


def read_geotiff_linear(path, band=1, is_db=True):
    """Read one band of a GeoTIFF and return (linear power, affine transform, crs).
    Requires rasterio. Earth Engine exports of COPERNICUS/S1_GRD are in dB."""
    import rasterio  # optional dependency
    with rasterio.open(path) as src:
        a = src.read(band).astype(np.float64)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        tr, crs = src.transform, src.crs
    return (10.0 ** (a / 10.0) if is_db else a), tr, crs


# ---------------------------------------------------------- synthetic scene
def synth_scene(shape=(1500, 1500), *, enl=4.4, n_ships=25, pixel_m=10.0,
                tcr_db=(12.0, 25.0), length_m=(90.0, 340.0), seed=0):
    """Simulated SAR scene: Gamma sea clutter with a wind gradient, one island,
    and `n_ships` rectangular targets. Returns (x, sea_mask, truth DataFrame).

    A unit test fixture, not a physical simulation: no sidelobes, no motion
    smear, no pixel correlation, no azimuth ambiguities.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    mean_db = -24.0 + 3.0 * np.sin(xx / 400.0) + 2.0 * yy / h      # sea state
    mu = 10.0 ** (mean_db / 10.0)
    x = rng.gamma(shape=enl, scale=mu / enl)

    # an island: bright, textured land that the sea mask removes
    land = ((xx - 0.78 * w) / (0.10 * w)) ** 2 + ((yy - 0.22 * h) / (0.07 * h)) ** 2 < 1.0
    x[land] = rng.gamma(shape=1.0, scale=10.0 ** (-8.0 / 10.0), size=int(land.sum()))
    sea = ~ndi.binary_dilation(land, iterations=int(round(300.0 / pixel_m)))

    truth, placed = [], np.zeros(shape, dtype=bool)
    tries = 0
    while len(truth) < n_ships and tries < 100 * n_ships:
        tries += 1
        length = rng.uniform(*length_m)
        beam = max(length / rng.uniform(5.5, 7.0), 15.0)
        theta = rng.uniform(0.0, np.pi)                              # heading
        r0, c0 = rng.uniform(60, h - 60), rng.uniform(60, w - 60)
        half = int(length / pixel_m / 2) + 3
        rs = slice(int(r0) - half, int(r0) + half + 1)
        cs = slice(int(c0) - half, int(c0) + half + 1)
        if rs.start < 0 or cs.start < 0 or rs.stop > h or cs.stop > w:
            continue
        ry, cx = np.mgrid[rs, cs]
        dy, dx = (ry - r0) * pixel_m, (cx - c0) * pixel_m
        along = dx * np.sin(theta) - dy * np.cos(theta)
        across = dx * np.cos(theta) + dy * np.sin(theta)
        hull = (np.abs(along) <= length / 2) & (np.abs(across) <= beam / 2)
        zone = ndi.binary_dilation(hull, iterations=8)
        if not sea[rs, cs][zone].all() or placed[rs, cs][zone].any():
            continue
        gain = 10.0 ** (rng.uniform(*tcr_db) / 10.0)
        patch = x[rs, cs]
        patch[hull] = rng.gamma(shape=1.0, scale=gain * mu[rs, cs][hull])
        placed[rs, cs] |= zone
        truth.append((r0, c0, length, beam, np.degrees(theta) % 180.0))
    truth = pd.DataFrame(truth, columns=["row", "col", "length_m", "beam_m", "heading_deg"])
    return x, sea, truth


def score_detections(det_df, truth_df, *, pixel_m=10.0, match_m=150.0):
    """Greedy one-to-one matching of detections to truth by distance."""
    if len(truth_df) == 0 or len(det_df) == 0:
        return {"tp": 0, "fp": len(det_df), "fn": len(truth_df),
                "precision": np.nan, "recall": np.nan, "pairs": []}
    d = np.hypot(det_df["row"].values[:, None] - truth_df["row"].values[None, :],
                 det_df["col"].values[:, None] - truth_df["col"].values[None, :]) * pixel_m
    pairs, used_d, used_t = [], set(), set()
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > match_m:
            break
        if i in used_d or j in used_t:
            continue
        used_d.add(i); used_t.add(j); pairs.append((int(i), int(j)))
    tp = len(pairs)
    fp, fn = len(det_df) - tp, len(truth_df) - tp
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan, "pairs": pairs}
