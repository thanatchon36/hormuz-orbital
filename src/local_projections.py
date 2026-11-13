"""
local_projections.py -- impulse responses by local projection (Jorda 2005).

For each horizon h = 0..H the regression is

    y_{t+h} - y_{t-1} = a_h + b_h * x_t
                        + sum_{l=1..p} ( phi_l * dy_{t-l} + psi_l * x_{t-l} )
                        + gamma' * controls_t + e_{t+h}

and b_h traces the cumulative response of y to a unit impulse in x. With y in
100 * log price and x a strike-exposure shock, b_h is the percent price response
h days after one unit of exposure.

Inference
    "hac"            Newey-West, bandwidth h + 1 (default)
    "lag_augmented"  heteroskedasticity-robust errors with p lags of y and x in
                     the regression (Montiel Olea and Plagborg-Moller 2021)

Monte Carlo check against a known impulse response in
tests/test_local_projections.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

__all__ = ["local_projection", "simulate_ar1_shock"]


def local_projection(y: pd.Series, x: pd.Series, *, controls: pd.DataFrame | None = None,
                     horizons: int = 20, lags: int = 5, ci: float = 0.90,
                     inference: str = "hac") -> pd.DataFrame:
    """Cumulative impulse response of y to x. Returns one row per horizon."""
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    dy = df["y"].diff()
    X = pd.DataFrame({"x": df["x"]})
    for l in range(1, lags + 1):
        X[f"dy_l{l}"] = dy.shift(l)
        X[f"x_l{l}"] = df["x"].shift(l)
    if controls is not None:
        X = X.join(controls.reindex(df.index))
    X = sm.add_constant(X)
    z = stats.norm.ppf(0.5 + ci / 2.0)

    rows = []
    for h in range(horizons + 1):
        dep = df["y"].shift(-h) - df["y"].shift(1)
        d = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        model = sm.OLS(d["dep"], d.drop(columns="dep"))
        if inference == "hac":
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        elif inference == "lag_augmented":
            fit = model.fit(cov_type="HC1")
        else:
            raise ValueError("inference must be 'hac' or 'lag_augmented'")
        b, se = fit.params["x"], fit.bse["x"]
        rows.append({"h": h, "beta": b, "se": se, "lo": b - z * se,
                     "hi": b + z * se, "nobs": int(fit.nobs)})
    return pd.DataFrame(rows)


def simulate_ar1_shock(T=1500, rho=0.8, theta=1.0, sigma_u=1.0, seed=0):
    """y_t = rho * y_{t-1} + theta * x_t + u_t with iid x. The response of
    y_{t+h} to x_t is theta * rho ** h."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    u = rng.normal(scale=sigma_u, size=T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = rho * y[t - 1] + theta * x[t] + u[t]
    idx = pd.RangeIndex(T)
    return pd.Series(y, idx), pd.Series(x, idx)
