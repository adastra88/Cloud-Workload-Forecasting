"""POST-HOC error analysis on the holdout.

Everything in this script is labelled post-hoc: it runs *after* the one-shot
scoring in ``scripts/final_score.py`` and it does **not** change the submitted
model or ``predictions.csv``.  Its purpose is to explain what the model got
wrong, not to fix it — fixing it would be retuning on the test set.

Produces: errors by horizon, peak vs non-peak (threshold from train), errors by
time of day of the window, worst timestamps, interval coverage under/over
confidence, and a comparison of the selected model against every baseline in the
same post-hoc table.
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
from cwf import metrics as M  # noqa: E402
from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import TARGETS, build_targets, infer_step_seconds  # noqa: E402

ART = ROOT / "reports" / "artifacts"
TARGET = "aggregate_cpu_utilization"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--predictions", default="predictions.csv")
    parser.add_argument("--holdout-len", type=int, default=24)
    args = parser.parse_args(argv)

    # Holdout truth read here too: allowed only because R7 has already run, and
    # every output of this script is labelled post-hoc.
    dataset = load_dataset(Path(args.data))
    series = build_targets(dataset.readings)
    step = infer_step_seconds(series["timestamp"])
    split = chronological_split(series, args.holdout_len, None)
    train, holdout = split.train, split.holdout
    preds = pd.read_csv(args.predictions)

    y = holdout[TARGET].to_numpy(dtype=float)
    yhat = preds[TARGET].to_numpy(dtype=float)
    ts = holdout["timestamp"].to_numpy(dtype=float)
    h = np.arange(1, len(y) + 1)
    err = y - yhat

    # ---- by horizon ------------------------------------------------------ #
    pd.DataFrame({
        "horizon_step": h,
        "minutes_ahead": h * step / 60.0,
        "actual": y,
        "predicted": yhat,
        "error_actual_minus_pred": err,
        "abs_error": np.abs(err),
    }).to_csv(ART / "posthoc_errors_by_horizon.csv", index=False)

    # ---- peak vs non-peak (threshold fixed on train only) ---------------- #
    peak_thr = float(np.quantile(train[TARGET].to_numpy(dtype=float), 0.90))
    is_peak = y >= peak_thr
    peak_rows = []
    for label, mask in (("peak_top_decile_train", is_peak), ("non_peak", ~is_peak)):
        if mask.sum() == 0:
            continue
        peak_rows.append({
            "segment": label,
            "n": int(mask.sum()),
            "threshold_pp": peak_thr,
            "mae": M.mae(y[mask], yhat[mask]),
            "rmse": M.rmse(y[mask], yhat[mask]),
            "bias": M.bias(y[mask], yhat[mask]),
            "mean_abs_actual_change_from_last_train_pp":
                float(np.mean(np.abs(y[mask] - train[TARGET].iloc[-1]))),
        })
    pd.DataFrame(peak_rows).to_csv(ART / "posthoc_peak_vs_nonpeak.csv", index=False)

    # ---- worst timestamps ------------------------------------------------ #
    order = np.argsort(-np.abs(err))[:5]
    pd.DataFrame({
        "timestamp": ts[order],
        "horizon_step": h[order],
        "actual": y[order],
        "predicted": yhat[order],
        "error": err[order],
    }).to_csv(ART / "posthoc_worst_steps.csv", index=False)

    # ---- interval coverage (post-hoc) ------------------------------------ #
    interval_path = ART / "forecast_intervals.csv"
    coverage_rows = []
    if interval_path.exists():
        band = pd.read_csv(interval_path)
        for tag in ("50", "80", "90", "95"):
            lo_col, hi_col = f"lower_{tag}", f"upper_{tag}"
            if lo_col not in band:
                continue
            lo = band[lo_col].to_numpy(dtype=float)
            hi = band[hi_col].to_numpy(dtype=float)
            coverage_rows.append({
                "nominal_level": float(tag) / 100.0,
                "coverage": M.coverage(y, lo, hi),
                "mean_width_pp": M.interval_width(lo, hi),
                "pinball_lower": M.pinball_loss(y, lo, 1 - float(tag) / 100.0),
                "pinball_upper": M.pinball_loss(y, hi, float(tag) / 100.0),
                "misses_below": int((y < lo).sum()),
                "misses_above": int((y > hi).sum()),
            })
        pd.DataFrame(coverage_rows).to_csv(ART / "posthoc_interval_coverage.csv", index=False)

    # ---- all candidates, post-hoc, same holdout -------------------------- #
    tr_values = train[TARGET].to_numpy(dtype=float)
    rows = [{"candidate": "submitted", "rmse": M.rmse(y, yhat), "mae": M.mae(y, yhat),
             "bias": M.bias(y, yhat)}]
    for name, fn in B.BASELINES.items():
        p = np.asarray(fn(tr_values, len(y)), dtype=float)
        rows.append({"candidate": f"baseline:{name}", "rmse": M.rmse(y, p), "mae": M.mae(y, p),
                     "bias": M.bias(y, p)})
    pd.DataFrame(rows).to_csv(ART / "posthoc_candidate_comparison.csv", index=False)

    # ---- churn context: a full-range look is now permitted --------------- #
    if dataset.metadata is not None:
        meta = dataset.metadata.copy()
        full_end = float(series["timestamp"].max())
        churn = pd.DataFrame([{
            "n_vms": int(len(meta)),
            "n_deleted_by_full_end": int((meta["timestamp_vm_deleted"] <= full_end).sum()),
            "n_deleted_in_holdout_window": int(
                ((meta["timestamp_vm_deleted"] > split.train_end_timestamp) &
                 (meta["timestamp_vm_deleted"] <= full_end)).sum()),
            "n_created_in_holdout_window": int(
                ((meta["timestamp_vm_created"] > split.train_end_timestamp) &
                 (meta["timestamp_vm_created"] <= full_end)).sum()),
        }])
        churn.to_csv(ART / "posthoc_churn_context.csv", index=False)

    summary = {
        "posthoc": True,
        "target": TARGET,
        "n_holdout": int(len(y)),
        "rmse": M.rmse(y, yhat),
        "mae": M.mae(y, yhat),
        "bias": M.bias(y, yhat),
        "holdout_mean_actual": float(np.mean(y)),
        "holdout_mean_predicted": float(np.mean(yhat)),
        "holdout_min": float(np.min(y)),
        "holdout_max": float(np.max(y)),
        "train_mean": float(np.mean(tr_values)),
        "train_mean_last_24": float(np.mean(tr_values[-24:])),
        "peak_threshold_train_pp": peak_thr,
        "lag24_autocorrelation_train": float(np.corrcoef(tr_values[:-24], tr_values[24:])[0, 1]),
    }
    (ART / "posthoc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
