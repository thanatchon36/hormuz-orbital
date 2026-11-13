"""Exposure index: full weight on an asset, none far away, lane term on the route."""
import pandas as pd

from strike_exposure import daily_exposure, exposure_index

INFRA = pd.DataFrame({"name": ["A", "B"], "lon": [50.0, 56.0], "lat": [26.0, 25.0],
                      "capacity_mboed": [6.0, 1.5], "hormuz_dependent": [1, 0]})
ROUTE = [[58.0, 25.0], [56.5, 26.6], [55.0, 26.1]]


def test_exposure_values():
    ev = pd.DataFrame({"date": ["2026-03-01", "2026-03-01", "2026-03-02"],
                       "lon": [50.0, 40.0, 56.5], "lat": [26.0, 35.0, 26.6]})
    ex = exposure_index(ev, INFRA, ROUTE, bandwidth_km=25, lane_bandwidth_km=40,
                        lane_weight_mboed=20)
    assert abs(ex["exposure_infra"].iloc[0] - 6.0) < 1e-6      # on asset A
    assert ex["exposure"].iloc[1] < 1e-6                       # 1,000+ km away
    assert abs(ex["exposure_lane"].iloc[2] - 20.0) < 1e-6      # on the route
    d = daily_exposure(ex)
    assert d.loc["2026-03-01", "n_events"] == 2
    assert abs(d.loc["2026-03-01", "shock"] - 6.0) < 1e-6
