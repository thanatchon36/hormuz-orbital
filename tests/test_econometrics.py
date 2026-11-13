"""Event study and local projections against known answers, plus one check
on the shipped real data."""
import numpy as np
import pandas as pd

from event_study import event_study, placebo_cars
from local_projections import local_projection, simulate_ar1_shock


def _price_with_jump(jump=0.10, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=600)
    r = rng.normal(0, 0.01, 600)
    r[400] += jump
    return pd.Series(70 * np.exp(np.cumsum(r)), idx), idx[400]


def test_event_study_recovers_a_known_jump():
    p, day0 = _price_with_jump()
    res, path = event_study(p, day0)
    assert abs(res["car_0_0"] - 10.0) < 2.5          # percent (log points)
    assert res["p_t_0_0"] < 0.001
    assert abs(float(path.loc[path["tau"] == -1, "car"].iloc[0])) < 1e-12


def test_placebo_distribution_is_centred():
    p, day0 = _price_with_jump(jump=0.0)
    pl = placebo_cars(p, window=(0, 5))
    assert abs(pl.mean()) < 0.5 and len(pl) > 400


def test_local_projection_recovers_ar1_response():
    rho, H = 0.8, 6
    est = np.zeros(H + 1)
    hit = np.zeros(H + 1)
    reps = 40
    for s in range(reps):
        y, x = simulate_ar1_shock(T=600, rho=rho, seed=s)
        irf = local_projection(y, x, horizons=H, lags=2)
        true = rho ** irf["h"].values
        est += irf["beta"].values / reps
        hit += ((irf["lo"] <= true) & (irf["hi"] >= true)).values / reps
    assert np.max(np.abs(est - rho ** np.arange(H + 1))) < 0.05
    assert hit.min() > 0.75


def test_real_data_abqaiq_day0():
    from data_sources import load_oil_prices
    res, _ = event_study(load_oil_prices()["brent"], "2019-09-16")
    assert 10.5 < res["car_0_0"] < 11.7              # Brent jumped about 11%
