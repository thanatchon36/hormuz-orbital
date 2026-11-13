"""
flow_index.py -- from satellite snapshot counts to a daily transit index.

1. Little's law.  A SAR image is a snapshot: it counts the vessels N that are
   inside the transit gate at one instant. For a gate of length G km crossed
   at mean speed v, each vessel spends W = G / v inside it, and in steady state

       E[N] = lambda * W      =>      lambda_hat = N * v / G     (transits/day)

   N is approximately Poisson, so Var[log lambda_hat] ~ 1 / N. This is the
   sampling-error budget of the index and it sets the gate length.

2. Fusion.  SAR passes are irregular and noisy; AIS is daily but under-reports
   when ships go dark. A linear Gaussian state-space model in logs combines
   them:

       state        m_t = log true flow          (random walk)
                    d_t = log AIS coverage       (0 and fixed in the calm
                                                  period, random walk after)
                    c   = log SAR efficiency     (constant, learned in calm)
       observation  log AIS_t = m_t + d_t + e_a
                    log SAR_t = m_t + c   + e_s  (only on days with a pass)

   In the calm period AIS is taken as truth (d = 0), which calibrates c. After
   the conflict start d_t is free, so SAR identifies the true flow and exp(d_t)
   is the estimated share of traffic that still broadcasts AIS.

The log transform is crude when daily counts are near zero. The production
version should use a Poisson observation model; this Gaussian version is the
transparent baseline. Validated on simulated data in tests/test_flow_index.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KN_TO_KM_PER_DAY = 1.852 * 24.0

__all__ = ["KN_TO_KM_PER_DAY", "haversine_km", "polyline_length_km", "route_length_in_box_km",
           "snapshot_to_flow", "expected_snapshot_count", "kalman_fuse",
           "simulate_fusion_case", "polygon_length_width_m", "detections_to_counts"]


# ----------------------------------------------------------------- geometry
def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Arguments broadcast."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))


def polyline_length_km(lonlat):
    p = np.asarray(lonlat, dtype=float)
    return float(haversine_km(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]).sum())


def route_length_in_box_km(route_lonlat, box, step_km=0.5):
    """Length of a route polyline that lies inside a lon/lat box.
    `box` = dict(lon_min, lon_max, lat_min, lat_max). Densifies the polyline."""
    p = np.asarray(route_lonlat, dtype=float)
    total = 0.0
    for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
        seg = float(haversine_km(x0, y0, x1, y1))
        n = max(int(np.ceil(seg / step_km)), 1)
        t = (np.arange(n) + 0.5) / n
        xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        inside = ((xs >= box["lon_min"]) & (xs <= box["lon_max"])
                  & (ys >= box["lat_min"]) & (ys <= box["lat_max"]))
        total += seg * inside.mean()
    return total


# ------------------------------------------------------------- Little's law
def snapshot_to_flow(n_vessels, gate_km, speed_kn=12.0, coverage=1.0):
    """Transits per day implied by a snapshot count (Little's law)."""
    n = np.asarray(n_vessels, dtype=float) / np.asarray(coverage, dtype=float)
    return n * speed_kn * KN_TO_KM_PER_DAY / gate_km


def expected_snapshot_count(flow_per_day, gate_km, speed_kn=12.0):
    """Expected vessels inside the gate at one instant, and the coefficient of
    variation of a single-snapshot flow estimate under Poisson counting."""
    n = np.asarray(flow_per_day, dtype=float) * gate_km / (speed_kn * KN_TO_KM_PER_DAY)
    with np.errstate(divide="ignore"):
        return n, 1.0 / np.sqrt(n)


# ------------------------------------------------------------ Kalman fusion
def _log_count(n):
    n = np.asarray(n, dtype=float)
    return np.log(n + 0.5), 1.0 / (n + 0.5)


def kalman_fuse(ais_count, sar_count, flow_per_vessel, conflict_start, *,
                q_m_calm=0.03 ** 2, q_m_conflict=0.10 ** 2, q_d=0.10 ** 2,
                break_var=4.0, r_ais_extra=0.10 ** 2, r_sar_extra=0.05 ** 2):
    """Fuse daily AIS counts with irregular SAR snapshot counts.

    Parameters
    ----------
    ais_count : pd.Series, daily AIS-derived transits (NaN = missing)
    sar_count : pd.Series on the same index, vessels counted inside the gate
                by each SAR pass (NaN on days without a pass)
    flow_per_vessel : float or Series, Little's-law factor v / G in transits
                per day per counted vessel; divide by the gate coverage of the
                scene where coverage is partial
    conflict_start : date from which AIS coverage d_t may drift; on that day
                both the flow and the coverage states receive a one-off
                innovation of variance `break_var` (a structural break)

    Returns a DataFrame with the smoothed true flow, its 90% band, the
    estimated AIS coverage share and the SAR efficiency constant.
    """
    idx = ais_count.index
    T = len(idx)
    start = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(conflict_start))))

    ya, _ = _log_count(ais_count.values)
    ya[~np.isfinite(ais_count.values)] = np.nan
    n_s = sar_count.reindex(idx).values.astype(float)
    kappa = (flow_per_vessel.reindex(idx).values.astype(float)
             if isinstance(flow_per_vessel, pd.Series) else float(flow_per_vessel))
    ys, _ = _log_count(n_s)
    ys = ys + np.log(kappa)
    ys[~np.isfinite(n_s)] = np.nan
    kap = np.broadcast_to(np.asarray(kappa, dtype=float), (T,))

    H_a, H_s = np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])
    first = ya[np.isfinite(ya)][0]
    a = np.array([first, 0.0, 0.0])
    P = np.diag([1.0, 1e-10, 1.0])

    a_f = np.zeros((T, 3)); P_f = np.zeros((T, 3, 3))
    a_p = np.zeros((T, 3)); P_p = np.zeros((T, 3, 3))
    loglik = 0.0
    for t in range(T):
        Q = np.diag([q_m_conflict if t >= start else q_m_calm,
                     q_d if t >= start else 0.0, 0.0])
        if t == start:
            Q[0, 0] += break_var
            Q[1, 1] += break_var
        a_pred, P_pred = a, P + Q
        a_p[t], P_p[t] = a_pred, P_pred
        a, P = a_pred, P_pred
        for y, H, extra, scale in ((ya[t], H_a, r_ais_extra, 1.0),
                                   (ys[t], H_s, r_sar_extra, kap[t])):
            if not np.isfinite(y):
                continue
            # Poisson variance of a log count, 1 / E[N], evaluated at the
            # PREDICTED count. Using the observed count would give more weight
            # to high draws and bias the index upward.
            r = 1.0 / max(np.exp(H @ a) / scale, 0.5) + extra
            f = H @ P @ H + r
            v = y - H @ a
            K = P @ H / f
            a = a + K * v
            P = P - np.outer(K, H @ P)
            loglik += -0.5 * (np.log(2 * np.pi * f) + v * v / f)
        a_f[t], P_f[t] = a, P

    # Rauch-Tung-Striebel smoother (transition matrix = identity)
    a_s, P_s = a_f.copy(), P_f.copy()
    for t in range(T - 2, -1, -1):
        J = P_f[t] @ np.linalg.pinv(P_p[t + 1])
        a_s[t] = a_f[t] + J @ (a_s[t + 1] - a_p[t + 1])
        P_s[t] = P_f[t] + J @ (P_s[t + 1] - P_p[t + 1]) @ J.T

    sd_m = np.sqrt(np.maximum(P_s[:, 0, 0], 0.0))
    out = pd.DataFrame({
        "flow": np.exp(a_s[:, 0]),
        "flow_lo": np.exp(a_s[:, 0] - 1.645 * sd_m),
        "flow_hi": np.exp(a_s[:, 0] + 1.645 * sd_m),
        "ais_coverage": np.exp(a_s[:, 1]),
        "sar_efficiency": np.exp(a_s[:, 2]),
    }, index=idx)
    out.attrs["loglik"] = float(loglik)
    return out


# -------------------------------------------------------------- simulation
def simulate_fusion_case(T=520, conflict_day=360, *, base_flow=50.0,
                         conflict_flow_share=0.30, ais_share_conflict=0.15,
                         gate_km=200.0, speed_kn=12.0, revisit_days=3,
                         sar_efficiency=0.9, seed=0):
    """Simulated chokepoint: true flow falls to `conflict_flow_share` of normal,
    while AIS reports only `ais_share_conflict` of what still transits."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-03-01", periods=T, freq="D")
    logf = np.log(base_flow) + np.cumsum(rng.normal(0, 0.02, T))
    logf -= np.linspace(0, logf[-1] - np.log(base_flow), T)          # no drift
    shock = np.zeros(T)
    k = np.arange(T - conflict_day)
    shock[conflict_day:] = np.log(conflict_flow_share) * (0.75 + 0.25 * np.exp(-k / 60.0))
    true = np.exp(logf + shock)
    cover = np.ones(T)
    cover[conflict_day:] = ais_share_conflict * np.exp(np.cumsum(rng.normal(0, 0.03, T - conflict_day)))
    cover = np.clip(cover, 0.02, 1.0)
    ais = rng.poisson(true * cover).astype(float)

    w_days = gate_km / (speed_kn * KN_TO_KM_PER_DAY)
    passes = np.zeros(T, dtype=bool)
    passes[rng.integers(0, revisit_days)::revisit_days] = True
    n_snap = np.where(passes, rng.poisson(true * w_days * sar_efficiency), np.nan)
    sar = snapshot_to_flow(n_snap, gate_km, speed_kn)
    return pd.DataFrame({"true_flow": true, "ais_true_coverage": cover,
                         "ais_count": ais, "sar_count": n_snap, "sar_flow": sar},
                        index=idx), idx[conflict_day]


# ------------------------------------------- Earth Engine exports -> counts
def polygon_length_width_m(geom):
    """Sides (m) of the minimum rotated rectangle of a small lon/lat polygon."""
    from shapely.affinity import affine_transform
    lat0 = geom.centroid.y
    kx, ky = 111_320.0 * np.cos(np.radians(lat0)), 110_540.0
    rect = affine_transform(geom, [kx, 0.0, 0.0, ky, 0.0, 0.0]).minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    e1 = float(np.hypot(xs[1] - xs[0], ys[1] - ys[0]))
    e2 = float(np.hypot(xs[2] - xs[1], ys[2] - ys[1]))
    return max(e1, e2), min(e1, e2)


def detections_to_counts(detections_geojson, coverage_csv, *, min_length_m=60.0,
                         max_length_m=480.0, tanker_min_length_m=180.0,
                         min_coverage=0.95, pass_gap_minutes=10):
    """Per-pass vessel counts from the two files that
    s1_gee_pipeline.export_detections() writes.

    Frames of the same pass (same platform and relative orbit, within
    `pass_gap_minutes`) are merged; passes that cover less than
    `min_coverage` of the area are dropped. Returns one row per pass with
    n_all (length within bounds) and n_large (length >= tanker_min_length_m).
    """
    import geopandas as gpd
    det = gpd.read_file(detections_geojson)
    if len(det):
        lw = [polygon_length_width_m(g) for g in det.geometry]
        det["length_m"] = [a for a, _ in lw]
        det["width_m"] = [b for _, b in lw]
        det = det[det["length_m"].between(min_length_m, max_length_m)]
        per_scene = det.groupby("scene_id").agg(
            n_all=("length_m", "size"),
            n_large=("length_m", lambda s: int((s >= tanker_min_length_m).sum())))
    else:
        per_scene = pd.DataFrame(columns=["n_all", "n_large"])
    cov = pd.read_csv(coverage_csv)
    cov["time"] = pd.to_datetime(cov["time_start"], unit="ms", utc=True)
    cov = cov.merge(per_scene, left_on="scene_id", right_index=True, how="left")
    cov[["n_all", "n_large"]] = cov[["n_all", "n_large"]].fillna(0)
    cov = cov.sort_values("time")
    new_pass = ((cov["time"].diff() > pd.Timedelta(minutes=pass_gap_minutes))
                | (cov["platform"] != cov["platform"].shift())
                | (cov["rel_orbit"] != cov["rel_orbit"].shift()))
    cov["pass_id"] = new_pass.cumsum()
    out = cov.groupby("pass_id").agg(time=("time", "min"), platform=("platform", "first"),
                                     orbit_pass=("orbit_pass", "first"),
                                     rel_orbit=("rel_orbit", "first"),
                                     coverage=("coverage", "sum"),
                                     n_all=("n_all", "sum"), n_large=("n_large", "sum"))
    out["coverage"] = out["coverage"].clip(upper=1.0)
    return out[out["coverage"] >= min_coverage].reset_index(drop=True)
