"""
strike_exposure.py -- from strike coordinates to an energy-exposure treatment.

For strike i at location s_i, with energy assets j of capacity c_j (mboe/d) at
locations a_j and the main shipping route R through the Strait:

    E_i = sum_j  c_j * exp( -d(s_i, a_j)^2 / (2 h^2) )
          + c_R * exp( -d(s_i, R)^2 / (2 h_R^2) )

d is great-circle distance in km, h and h_R are kernel bandwidths, and c_R is
the flow through the Strait. E_i is measured in mboe/d of capacity "within
reach" of the strike. It is an exposure measure, not a damage estimate: a
strike that lands on a refinery and one that is intercepted above it score the
same. Damage is measured separately from imagery (thermal anomalies, burn
scars, SAR coherence loss).

The daily shock used in the regressions is

    S_t = sum over strikes on day t of  w_i * E_i

with w_i = 1 by default, or a severity weight such as log(1 + munitions).

Bandwidths and c_R are tuning parameters. The thesis reports results on a grid
of values and chooses none of them by looking at price responses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from flow_index import haversine_km

__all__ = ["densify", "distance_to_route_km", "exposure_index", "daily_exposure"]


def densify(route_lonlat, step_km: float = 2.0) -> np.ndarray:
    """Insert points along a polyline so that spacing is at most `step_km`."""
    p = np.asarray(route_lonlat, dtype=float)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(np.ceil(float(haversine_km(a[0], a[1], b[0], b[1])) / step_km)), 1)
        t = np.arange(1, n + 1) / n
        out.extend(np.column_stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]))
    return np.asarray(out)


def distance_to_route_km(lon, lat, route_lonlat) -> np.ndarray:
    """Distance from each point to the nearest point of the (densified) route."""
    r = densify(route_lonlat)
    lon, lat = np.atleast_1d(lon).astype(float), np.atleast_1d(lat).astype(float)
    d = haversine_km(lon[:, None], lat[:, None], r[None, :, 0], r[None, :, 1])
    return d.min(axis=1)


def exposure_index(events: pd.DataFrame, infra: pd.DataFrame, route_lonlat=None, *,
                   bandwidth_km: float = 25.0, lane_bandwidth_km: float = 40.0,
                   lane_weight_mboed: float = 20.0, hormuz_only: bool = False) -> pd.DataFrame:
    """Exposure of each event. `events` needs lon, lat; `infra` needs lon, lat,
    capacity_mboed. Returns events with columns exposure_infra, exposure_lane,
    exposure and nearest_asset / nearest_km."""
    ev = events.copy()
    inf = infra[infra["hormuz_dependent"] == 1] if hormuz_only else infra
    d = haversine_km(ev["lon"].values[:, None], ev["lat"].values[:, None],
                     inf["lon"].values[None, :], inf["lat"].values[None, :])
    k = np.exp(-0.5 * (d / bandwidth_km) ** 2)
    ev["exposure_infra"] = k @ inf["capacity_mboed"].values
    j = d.argmin(axis=1)
    ev["nearest_asset"] = inf["name"].values[j]
    ev["nearest_km"] = d[np.arange(len(ev)), j]
    if route_lonlat is not None:
        dr = distance_to_route_km(ev["lon"].values, ev["lat"].values, route_lonlat)
        ev["route_km"] = dr
        ev["exposure_lane"] = lane_weight_mboed * np.exp(-0.5 * (dr / lane_bandwidth_km) ** 2)
    else:
        ev["exposure_lane"] = 0.0
    ev["exposure"] = ev["exposure_infra"] + ev["exposure_lane"]
    return ev


def daily_exposure(events: pd.DataFrame, *, date_col: str = "date",
                   weight_col: str | None = None, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Aggregate event exposure to a daily shock series S_t.

    Returns columns: shock (sum of weighted exposure), n_events,
    max_exposure. Days without events are zero when `index` is given.
    """
    w = events[weight_col] if weight_col else 1.0
    df = pd.DataFrame({"date": pd.to_datetime(events[date_col]).dt.normalize(),
                       "wx": events["exposure"] * w, "x": events["exposure"]})
    g = df.groupby("date")
    out = pd.DataFrame({"shock": g["wx"].sum(), "n_events": g.size(),
                        "max_exposure": g["x"].max()})
    if index is not None:
        out = out.reindex(index).fillna(0.0)
    return out
