"""Little's law and the SAR-AIS Kalman fusion on simulated data."""
import numpy as np

from flow_index import (KN_TO_KM_PER_DAY, expected_snapshot_count, kalman_fuse,
                        route_length_in_box_km, simulate_fusion_case, snapshot_to_flow)


def test_littles_law_round_trip():
    n, cv = expected_snapshot_count(88.5, 200.0, 12.0)
    assert abs(float(snapshot_to_flow(n, 200.0, 12.0)) - 88.5) < 1e-9
    assert abs(cv - 1 / np.sqrt(n)) < 1e-12


def test_route_length_in_box():
    route = [[0.0, 0.0], [2.0, 0.0]]
    box = dict(lon_min=0.5, lon_max=1.5, lat_min=-1, lat_max=1)
    assert abs(route_length_in_box_km(route, box) - 111.19) < 1.0


def test_fusion_recovers_true_flow_when_ais_goes_dark():
    gate, v = 200.0, 12.0
    kappa = v * KN_TO_KM_PER_DAY / gate
    bias, ais_bias, cover_err, inside = [], [], [], []
    for seed in range(12):
        sim, start = simulate_fusion_case(seed=seed, gate_km=gate, speed_kn=v)
        out = kalman_fuse(sim["ais_count"], sim["sar_count"], kappa, start)
        c = sim.index >= start
        bias.append(out["flow"][c].mean() / sim["true_flow"][c].mean() - 1)
        ais_bias.append(sim["ais_count"][c].mean() / sim["true_flow"][c].mean() - 1)
        cover_err.append(out["ais_coverage"][c].mean() - sim["ais_true_coverage"][c].mean())
        inside.append(((sim["true_flow"] >= out["flow_lo"]) & (sim["true_flow"] <= out["flow_hi"]))[c].mean())
    assert np.mean(ais_bias) < -0.8                 # AIS alone is badly biased
    assert abs(np.mean(bias)) < 0.05                # fused index is not
    assert abs(np.mean(cover_err)) < 0.03
    assert np.mean(inside) > 0.85                   # 90% band is honest


def test_fusion_does_not_invent_dark_traffic():
    gate, v = 200.0, 12.0
    kappa = v * KN_TO_KM_PER_DAY / gate
    est = []
    for seed in range(8):
        sim, start = simulate_fusion_case(seed=seed, ais_share_conflict=1.0)
        out = kalman_fuse(sim["ais_count"], sim["sar_count"], kappa, start)
        c = sim.index >= start
        est.append(out["ais_coverage"][c].mean() - sim["ais_true_coverage"][c].mean())
    assert abs(np.mean(est)) < 0.05
