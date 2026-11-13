"""
event_study.py -- abnormal returns of an energy price around strike dates.

Model.  Daily log returns r_t (in percent). The normal return is the mean of r
over an estimation window that ends before the event (constant-mean model,
the standard choice for a single commodity with no market factor):

    AR_t        = r_t - mean(r | estimation window)
    CAR(a, b)   = sum of AR_t for t = a..b      (trading days, 0 = event day)

Inference, two ways, because the number of events is small:

  * parametric: t = CAR / sqrt(L * s^2 * (1 + L / n)), L = b - a + 1,
    s^2 and n from the estimation window, Student-t with n - 1 d.o.f.
  * placebo: the same CAR is computed at every trading day of a long reference
    sample that does not overlap an event window; the p-value is the share of
    placebo CARs at least as extreme as the observed one.

Day 0 must be the first trading day whose close postdates the event. Weekend
events therefore map to the following Monday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["log_returns", "event_study", "placebo_cars", "run_event_table"]


def log_returns(price: pd.Series) -> pd.Series:
    """100 * log return; non-positive prices become missing."""
    p = price.where(price > 0).dropna()
    return (100.0 * np.log(p).diff()).dropna()


def _locate(index: pd.DatetimeIndex, day0) -> int:
    i = int(index.searchsorted(pd.Timestamp(day0), side="left"))
    if i >= len(index):
        raise ValueError(f"day0 {day0} is after the end of the sample")
    return i


def event_study(price: pd.Series, day0, *, est=(-130, -11), win=(-5, 20),
                car_windows=((0, 0), (0, 1), (0, 5), (0, 10), (0, 20))):
    """Event study for one event. Returns (summary dict, AR path DataFrame)."""
    r = log_returns(price)
    i = _locate(r.index, day0)
    if i + est[0] < 0:
        raise ValueError("not enough history for the estimation window")
    est_r = r.iloc[i + est[0]: i + est[1] + 1]
    n, mu, s = len(est_r), est_r.mean(), est_r.std(ddof=1)

    lo, hi = win[0], min(win[1], len(r) - 1 - i)
    ar = r.iloc[i + lo: i + hi + 1] - mu
    path = pd.DataFrame({"tau": np.arange(lo, hi + 1), "date": ar.index,
                         "ar": ar.values})
    # cumulative path normalised to 0 at tau = -1, so pre-trends are visible
    c = path["ar"].cumsum()
    base = float(c[path["tau"] == -1].iloc[0]) if (path["tau"] == -1).any() else 0.0
    path["car"] = c - base

    out = {"day0": r.index[i], "n_est": n, "mean_est": mu, "sd_est": s}
    for a, b in car_windows:
        if b > hi:
            out[f"car_{a}_{b}"] = np.nan
            out[f"t_{a}_{b}"] = np.nan
            out[f"p_t_{a}_{b}"] = np.nan
            continue
        L = b - a + 1
        car = float(path.loc[(path["tau"] >= a) & (path["tau"] <= b), "ar"].sum())
        se = np.sqrt(L * s * s * (1.0 + L / n))
        t = car / se
        out[f"car_{a}_{b}"] = car
        out[f"t_{a}_{b}"] = t
        out[f"p_t_{a}_{b}"] = float(2.0 * stats.t.sf(abs(t), df=n - 1))
    return out, path


def placebo_cars(price: pd.Series, *, est=(-130, -11), window=(0, 5),
                 start=None, end=None, exclude=(), exclude_halfwidth=30):
    """CAR(window) at every admissible trading day (vectorised).

    `exclude` is a list of event dates; days within `exclude_halfwidth`
    trading days of any of them are dropped from the placebo sample.
    """
    r = log_returns(price)
    a, b = window
    L = b - a + 1
    n_est = est[1] - est[0] + 1
    csum = np.concatenate([[0.0], np.cumsum(r.values)])
    T = len(r)
    idx = np.arange(T)
    ok = (idx + est[0] >= 0) & (idx + b < T)
    i = idx[ok]
    win_sum = csum[i + b + 1] - csum[i + a]
    est_mean = (csum[i + est[1] + 1] - csum[i + est[0]]) / n_est
    car = pd.Series(win_sum - L * est_mean, index=r.index[i])
    if start is not None:
        car = car[car.index >= pd.Timestamp(start)]
    if end is not None:
        car = car[car.index <= pd.Timestamp(end)]
    if len(exclude):
        pos = np.array([_locate(r.index, d) for d in exclude])
        loc = r.index.get_indexer(car.index)
        near = (np.abs(loc[:, None] - pos[None, :]) <= exclude_halfwidth).any(axis=1)
        car = car[~near]
    return car


def run_event_table(price: pd.Series, events: pd.DataFrame, *, est=(-130, -11),
                    win=(-5, 20),
                    car_windows=((0, 0), (0, 1), (0, 5), (0, 10), (0, 20)),
                    placebo_start="2015-01-01", id_col="episode_id",
                    day0_col="day0"):
    """Event study for every row of `events`, with placebo p-values.

    Returns (table, paths): one row per event, and the stacked AR/CAR paths.
    """
    rows, paths = [], []
    day0s = list(events[day0_col])
    placebo = {w: placebo_cars(price, est=est, window=w, start=placebo_start,
                               exclude=day0s) for w in car_windows}
    for _, ev in events.iterrows():
        res, path = event_study(price, ev[day0_col], est=est, win=win,
                                car_windows=car_windows)
        for (a, b) in car_windows:
            car = res[f"car_{a}_{b}"]
            pl = placebo[(a, b)].values
            res[f"p_placebo_{a}_{b}"] = (float((np.abs(pl) >= abs(car)).mean())
                                         if np.isfinite(car) else np.nan)
        res[id_col] = ev[id_col]
        rows.append(res)
        path[id_col] = ev[id_col]
        paths.append(path)
    table = pd.DataFrame(rows).set_index(id_col)
    table.attrs["n_placebo"] = {f"{a}_{b}": int(len(v)) for (a, b), v in placebo.items()}
    return table, pd.concat(paths, ignore_index=True)
