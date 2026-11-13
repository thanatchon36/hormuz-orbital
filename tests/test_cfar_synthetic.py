"""CFAR detector against simulated Gamma clutter and injected targets."""
import numpy as np

from s1_cfar import (cfar_detect, detect_vessels, gamma_threshold_multiplier,
                     score_detections, synth_scene)


def test_threshold_multiplier_is_a_gamma_quantile():
    rng = np.random.default_rng(0)
    x = rng.gamma(4.4, 1 / 4.4, size=2_000_000)
    t = gamma_threshold_multiplier(4.4, 1e-3)
    assert abs((x > t).mean() - 1e-3) < 2e-4


def test_false_alarm_rate_matches_design_on_pure_clutter():
    rng = np.random.default_rng(1)
    x = rng.gamma(4.4, 1 / 4.4, size=(2000, 2000))
    det, _ = cfar_detect(x, pfa=1e-4, pfa_grow=None, n_iter=1)
    inner = det[130:-130, 130:-130]
    assert 0.5e-4 < inner.mean() < 2e-4


def test_no_objects_on_pure_clutter_at_default_settings():
    rng = np.random.default_rng(2)
    x = rng.gamma(4.4, 1 / 4.4, size=(2000, 2000))
    assert len(detect_vessels(x)) == 0


def test_precision_recall_and_length_on_synthetic_scenes():
    tp = fp = fn = 0
    err = []
    for seed in range(6):
        x, sea, truth = synth_scene(seed=seed, n_ships=25)
        det = detect_vessels(x, sea)
        s = score_detections(det, truth)
        tp, fp, fn = tp + s["tp"], fp + s["fp"], fn + s["fn"]
        err += [det["length_m"].values[i] - truth["length_m"].values[j] for i, j in s["pairs"]]
    assert tp / (tp + fp) > 0.97
    assert tp / (tp + fn) > 0.97
    assert abs(np.mean(err)) < 10 and np.std(err) < 25      # metres


def test_land_is_ignored():
    x, sea, _ = synth_scene(seed=0, n_ships=0)
    det = detect_vessels(x, sea)
    assert len(det) == 0
