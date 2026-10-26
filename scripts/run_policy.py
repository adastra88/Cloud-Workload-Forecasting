"""Calibrate the prediction band, choose the provisioning level, and save both.

Everything here is fitted on rolling-origin folds **inside the training window**.
The holdout is not read; `scripts/final_score.py` scores the frozen result once.

Outputs
-------
* ``reports/artifacts/forecast_intervals.csv``  band around the submitted forecast
* ``reports/artifacts/interval_calibration.csv`` raw vs conformal vs widened coverage
* ``reports/artifacts/policy_selection.csv``     net cost per provisioning level
* ``reports/artifacts/policy_sensitivity.csv``   cost-ratio sensitivity
* ``reports/artifacts/policy_choice.json``       the frozen margin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwf import baselines as B  # noqa: E402
from cwf import evaluate as E  # noqa: E402
from cwf import intervals as I  # noqa: E402
from cwf import metrics as M  # noqa: E402
from cwf import models as MO  # noqa: E402
from cwf import policy as P  # noqa: E402
from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import TARGETS, build_targets, infer_step_seconds  # noqa: E402

ART = ROOT / "reports" / "artifacts"
CONFIG = ROOT / "config" / "final_model.json"


def get_model(name: str, alpha: float = 10.0):
    if name in B.BASELINES:
        return B.BASELINES[name]
    return MO.candidate_models(alpha=alpha).get(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--holdout-len", type=int, default=24)
    parser.add_argument("--c-under", type=float, default=P.DEFAULT_C_UNDER)
    parser.add_argument("--c-over", type=float, default=P.DEFAULT_C_OVER)
    args = parser.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}

    dataset = load_dataset(Path(args.data))
    series = build_targets(dataset.readings)
    step = infer_step_seconds(series["timestamp"])
    split = chronological_split(series, args.holdout_len, None)
    train = split.train[["timestamp", *TARGETS]].reset_index(drop=True)
    n = len(train)

    folds = E.make_folds(n, horizon=args.holdout_len, min_train=E.DEFAULT_MIN_TRAIN, step=E.DEFAULT_STEP)
    target = "aggregate_cpu_utilization"
    name = config.get("targets", {}).get(target, {}).get("model", "persistence")
    model = get_model(name, float(config.get("ridge_alpha", 10.0)))
    values = train[target].to_numpy(dtype=float)
    res = E.evaluate_candidate(model, values, folds, target, name=name)

    # ---- interval calibration ------------------------------------------- #
    calib_rows = []
    best = None
    for method in ("empirical", "conformal"):
        q = I.quantiles_by_horizon(res.residuals, levels=I.DEFAULT_LEVELS, method=method)
        for level in (0.50, 0.80, 0.90, 0.95):
            lo, hi = I.band(res.y_pred.ravel(), q, level)
            cov = M.coverage(res.y_true.ravel(), lo, hi)
            # leave-one-origin-out, honest coverage
            loyo = I.leave_one_origin_out_coverage(res.y_true, res.y_pred, level)
            width = M.interval_width(lo, hi)
            calib_rows.append({
                "method": method, "level": level, "coverage_in_sample": cov,
                "coverage_loyo": loyo, "mean_width_pp": width,
                "width_ratio_vs_90": float("nan"),
            })
            if level == 0.90:
                score = abs((loyo if np.isfinite(loyo) else cov) - 0.90)
                if best is None or score < best[0]:
                    best = (score, method, level)

    # Widening factor: inflate the 90% residual quantiles until LOYO coverage
    # reaches nominal on the folds.  Calibrated on training folds only.
    method = best[1] if best else "empirical"
    q90 = I.quantiles_by_horizon(res.residuals, levels=I.DEFAULT_LEVELS, method=method)
    chosen_factor = 1.0
    for factor in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
        hits = []
        for i in range(res.y_true.shape[0]):
            mask = np.arange(res.y_true.shape[0]) != i
            q_i = I.quantiles_by_horizon(I.residuals_by_horizon(res.y_true[mask], res.y_pred[mask]),
                                         levels=I.DEFAULT_LEVELS, method=method)
            lo, hi = I.band(res.y_pred[i], q_i, 0.90)
            lo, hi = I.widen(lo, hi, factor)
            hits.extend(((res.y_true[i] >= lo) & (res.y_true[i] <= hi)).tolist())
        cov = float(np.mean(hits))
        calib_rows.append({"method": f"widen_x{factor}", "level": 0.90, "coverage_in_sample": float("nan"),
                           "coverage_loyo": cov, "mean_width_pp": float("nan"),
                           "width_ratio_vs_90": float("nan")})
        if cov >= 0.90 and chosen_factor == 1.0 and factor > 1.0:
            chosen_factor = factor
    pd.DataFrame(calib_rows).to_csv(ART / "interval_calibration.csv", index=False)

    # ---- forecast the holdout timestamps and attach the band ------------- #
    holdout_ts = split.holdout["timestamp"].to_numpy()
    fn = model.forecast if hasattr(model, "forecast") else model
    point = np.asarray(fn(values, len(holdout_ts)), dtype=float).ravel()
    point = np.clip(point, 0.0, 100.0)

    def build_band(level: float):
        lo, hi = I.band(point, q90, level)
        lo, hi = I.widen(lo, hi, chosen_factor)
        return np.clip(lo, 0.0, 100.0), np.clip(hi, 0.0, 100.0)

    bands = {level: build_band(level) for level in (0.50, 0.80, 0.90, 0.95)}
    band_90 = bands[0.90]

    interval_frame = pd.DataFrame({
        "timestamp": holdout_ts,
        "horizon_step": np.arange(1, len(holdout_ts) + 1),
        "forecast_aggregate_cpu_utilization": point,
    })
    for level, (lo, hi) in bands.items():
        tag = f"{int(round(level * 100))}"
        interval_frame[f"lower_{tag}"] = lo
        interval_frame[f"upper_{tag}"] = hi
    interval_frame["model"] = name
    interval_frame["widen_factor"] = chosen_factor
    interval_frame.to_csv(ART / "forecast_intervals.csv", index=False)

    # ---- provisioning level on the training folds ------------------------ #
    fold_payload = []
    for f in res.results:
        fwd = np.asarray(f.y_pred, dtype=float)
        lo, hi = I.band(fwd, q90, 0.90)
        lo, hi = I.widen(lo, hi, chosen_factor)
        fold_payload.append({"forecast": fwd, "lower": lo, "upper": hi, "realised": np.asarray(f.y_true, float)})

    best_level, policy_rows = P.choose_level(fold_payload, c_under=args.c_under, c_over=args.c_over)
    pd.DataFrame(policy_rows).assign(chosen=False).to_csv(ART / "policy_selection.csv", index=False)

    prov_by_level = {level: P.level_to_band(np.concatenate([f["forecast"] for f in fold_payload]),
                                            np.concatenate([f["lower"] for f in fold_payload]),
                                            np.concatenate([f["upper"] for f in fold_payload]), level)
                     for level in P.DEFAULT_LEVELS}
    realised = np.concatenate([f["realised"] for f in fold_payload])
    sens = pd.DataFrame(P.sensitivity(prov_by_level, realised))
    sens.to_csv(ART / "policy_sensitivity.csv", index=False)

    bench_width = float(np.median(np.abs(np.diff(values)))) if values.size > 1 else 0.0
    lo90, hi90 = bands[0.90]
    peak_threshold = float(np.quantile(values, 0.90))
    trigger = P.peak_risk_trigger(point, lo90, hi90, peak_threshold, bench_width=bench_width)

    choice = {
        "target": target,
        "model": name,
        "c_under": args.c_under,
        "c_over": args.c_over,
        "newsvendor_level": P.newsvendor_quantile(args.c_under, args.c_over),
        "chosen_level": best_level,
        "interval_method": method,
        "widen_factor": chosen_factor,
        "peak_threshold_train_pp": peak_threshold,
        "bench_width_median_abs_diff_pp": bench_width,
        "peak_risk_on_holdout_band": trigger,
        "note": ("Level and margin chosen on rolling-origin folds inside the training window; "
                 "the holdout is scored once by scripts/final_score.py."),
    }
    (ART / "policy_choice.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps({k: choice[k] for k in ("model", "newsvendor_level", "chosen_level",
                                             "interval_method", "widen_factor")}, indent=2))
    print(f"peak trigger: {trigger['action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
