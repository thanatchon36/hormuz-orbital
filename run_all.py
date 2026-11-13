"""
run_all.py -- reproduce every table and figure of the research design.

    python fetch_basemap.py      # once, for the maps
    python run_all.py            # tables in outputs/tables, figures in outputs/figures
    python run_all.py --refresh  # re-download prices and PortWatch first

Everything here runs on real, public data: EIA Brent and WTI spot prices and
IMF PortWatch daily transit counts for the Strait of Hormuz. Nothing is
simulated except the two demo figures, which say so in their titles.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from data_sources import load_oil_prices, load_portwatch, load_reference   # noqa: E402
from event_study import run_event_table                                    # noqa: E402
from flow_index import expected_snapshot_count, route_length_in_box_km      # noqa: E402
from local_projections import local_projection                             # noqa: E402
from strike_exposure import exposure_index                                 # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
TAB = ROOT / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def flow_response(pw, events, col="n_tanker"):
    """Change in mean daily transits: days +1..+7 and +8..+28 after the event
    against a baseline of days -35..-8."""
    rows = []
    for _, e in events.iterrows():
        d = e["calendar_date"]
        base = pw.loc[d - pd.Timedelta(days=35): d - pd.Timedelta(days=8), col]
        if len(base) < 20:
            rows.append({"episode_id": e["episode_id"]})
            continue
        w1 = pw.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=7), col]
        w2 = pw.loc[d + pd.Timedelta(days=8): d + pd.Timedelta(days=28), col]
        rows.append({"episode_id": e["episode_id"], "baseline_per_day": base.mean(),
                     "week1_per_day": w1.mean(), "weeks2_4_per_day": w2.mean(),
                     "flow_chg_week1_pct": 100 * (w1.mean() / base.mean() - 1),
                     "flow_chg_weeks2_4_pct": 100 * (w2.mean() / base.mean() - 1)})
    return pd.DataFrame(rows).set_index("episode_id")


def flow_noise_band(pw, calm, col="n_tanker"):
    """Distribution of the same week-1 statistic on every calm-period day."""
    s = pw.loc[calm[0]:calm[1], col]
    week1 = s.rolling(7).mean().shift(-7)
    base = s.rolling(28).mean().shift(8)
    r = (100 * (week1 / base - 1)).dropna()
    return {"sd_pct": float(r.std()), "p05_pct": float(r.quantile(0.05)),
            "p95_pct": float(r.quantile(0.95)), "n": int(len(r))}


def main(refresh=False):
    px = load_oil_prices(refresh=refresh)
    pw = load_portwatch("hormuz", refresh=refresh)
    ev = load_reference("episode_onsets")
    es, lp = CFG["event_study"], CFG["local_projections"]
    summary = {"brent_last_date": str(px["brent"].dropna().index.max().date()),
               "portwatch_last_date": str(pw.index.max().date())}

    # 1. event study on Brent ------------------------------------------------
    table, paths = run_event_table(px["brent"], ev, est=tuple(es["estimation_window"]),
                                   win=tuple(es["event_window"]),
                                   car_windows=tuple(map(tuple, es["car_windows"])),
                                   placebo_start=es["placebo_start"])
    summary["placebo_days"] = table.attrs.get("n_placebo", {})
    table = ev.set_index("episode_id")[["label", "prior_exposure"]].join(table)
    table.to_csv(TAB / "event_study_brent.csv")
    paths.to_csv(TAB / "event_study_paths.csv", index=False)

    # 2. transit response per episode ---------------------------------------
    fr = flow_response(pw, ev)
    fr.to_csv(TAB / "flow_response.csv")
    summary["flow_noise_band_week1"] = flow_noise_band(pw, CFG["flow"]["calm_period"])

    # 3. Brent minus WTI spread around each episode ---------------------------
    sp = (px["brent"] - px["wti"]).dropna()
    rows = []
    for _, e in ev.iterrows():
        i = sp.index.searchsorted(e["day0"])
        rows.append({"episode_id": e["episode_id"], "spread_day_m1": sp.iloc[i - 1],
                     "spread_day_p20": sp.iloc[i + 20],
                     "spread_chg_usd": sp.iloc[i + 20] - sp.iloc[i - 1]})
    pd.DataFrame(rows).set_index("episode_id").to_csv(TAB / "spread_response.csv")

    # 4. calm-period local projection: does ordinary transit noise move Brent?
    wk = pd.DataFrame({"y": 100 * np.log(px["brent"].resample("W-FRI").last()),
                       "f": 100 * np.log(pw["n_tanker"].resample("W-FRI").sum())}).dropna()
    calm = wk.loc[CFG["flow"]["calm_period"][0]: CFG["flow"]["calm_period"][1]].iloc[1:]
    irf = local_projection(calm["y"], -calm["f"].diff(), horizons=6, lags=4, ci=lp["ci"])
    irf.to_csv(TAB / "lp_calm_weekly.csv", index=False)
    summary["lp_calm"] = {"weeks": int(len(calm)),
                          "corr_dlog_flow_dlog_brent": float(calm["f"].diff().corr(calm["y"].diff())),
                          "beta_h0_per_10pct_fall": float(10 * irf.loc[0, "beta"]),
                          "ci_h0_per_10pct_fall": [float(10 * irf.loc[0, "lo"]), float(10 * irf.loc[0, "hi"])]}

    # 5. inside the closure: weekly transits against Brent --------------------
    w = pd.DataFrame({"tankers_per_day": pw["n_tanker"].resample("W-FRI").mean(),
                      "brent": px["brent"].resample("W-FRI").last()}).dropna()
    clo = w.loc["2026-03-06": pw.index.max()]          # full weeks only
    clo.to_csv(TAB / "closure_weekly.csv")
    summary["closure"] = {"weeks": int(len(clo)),
                          "corr_levels": float(clo["brent"].corr(clo["tankers_per_day"])),
                          "corr_changes": float(clo["brent"].diff().corr(clo["tankers_per_day"].diff()))}

    # 6. headline levels ------------------------------------------------------
    pre = slice("2025-03-01", "2026-02-28")
    post = slice("2026-03-01", str(pw.index.max().date()))
    summary["levels"] = {
        "tankers_pre": float(pw.loc[pre, "n_tanker"].mean()),
        "tankers_post": float(pw.loc[post, "n_tanker"].mean()),
        "all_pre": float(pw.loc[pre, "n_total"].mean()),
        "all_post": float(pw.loc[post, "n_total"].mean()),
        "brent_pre": float(px["brent"].loc["2025-03-01":"2026-02-27"].mean()),
        "brent_post": float(px["brent"].loc["2026-03-01":"2026-10-31"].mean()),
        "brent_27feb": float(px["brent"].loc["2026-02-27"]),
        "brent_peak": float(px["brent"].loc["2026-03-01":].max()),
        "brent_peak_date": str(px["brent"].loc["2026-03-01":].idxmax().date()),
        "spread_pre": float(sp.loc["2025-03-01":"2026-02-27"].mean()),
        "spread_post": float(sp.loc["2026-03-01":"2026-10-31"].mean())}

    # 7. exposure index on the illustrative reference layer -------------------
    ex = exposure_index(load_reference("strike_events_reference"),
                        load_reference("energy_infrastructure"), CFG["route_schematic"],
                        bandwidth_km=CFG["exposure"]["bandwidth_km"],
                        lane_bandwidth_km=CFG["exposure"]["lane_bandwidth_km"],
                        lane_weight_mboed=CFG["exposure"]["lane_weight_mboed"])
    ex.to_csv(TAB / "exposure_reference_layer.csv", index=False)

    # 8. sampling-error budget of the SAR gate --------------------------------
    gate_km = route_length_in_box_km(CFG["route_schematic"], CFG["aoi"]["gate"])
    n, cv = expected_snapshot_count(float(pw.loc[CFG["flow"]["calm_period"][0]:CFG["flow"]["calm_period"][1], "n_total"].mean()),
                                    gate_km, CFG["flow"]["mean_speed_kn"])
    summary["gate"] = {"route_km_in_gate": float(gate_km), "expected_vessels": float(n),
                       "single_snapshot_cv": float(cv)}

    json.dump(summary, open(TAB / "summary.json", "w"), indent=2)

    # 9. figures --------------------------------------------------------------
    import make_figures as mf
    made = []
    for f in (mf.fig_regional_map, mf.fig_hormuz_map, mf.fig_evidence,
              mf.fig_cfar_demo, mf.fig_fusion_demo):
        try:
            made.append(str(f().relative_to(ROOT)))
        except FileNotFoundError as err:
            print("skipped", f.__name__, "- run fetch_basemap.py first:", err)

    pd.set_option("display.width", 200)
    cols = ["label"] + [c for c in table.columns if c.startswith("car_")]
    print(table[cols].round(1).to_string())
    print(fr.round(1).to_string())
    print(json.dumps(summary, indent=2))
    print("figures:", *made, sep="\n  ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download prices and PortWatch")
    main(ap.parse_args().refresh)
