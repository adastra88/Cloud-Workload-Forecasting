"""Phase 5: apply the pre-registered selection rule and freeze the model config.

Reads the rolling-origin results produced by ``run_validation.py`` (or recomputes
them), applies the 1-SE simplicity rule from docs/protocol.md §7, writes
``config/final_model.json``, and saves the residual quantiles used for the
prediction intervals.

Uses the training window only.  No holdout value is read here.
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
from cwf import models as MO  # noqa: E402
from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import TARGETS, build_targets  # noqa: E402

ART = ROOT / "reports" / "artifacts"
CONFIG = ROOT / "config" / "final_model.json"

COMPLEXITY_ORDER = [
    "persistence", "ma6", "ma12", "ma24", "drift", "seasonal_naive12",
    "ets_holt", "ets_damped", "ridge_direct", "gbm_direct",
]
LEVELS = I.DEFAULT_LEVELS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--holdout-len", type=int, default=24)
    parser.add_argument("--train-len", type=int, default=110)
    args = parser.parse_args(argv)

    dataset = load_dataset(Path(args.data))
    series = build_targets(dataset.readings)
    split = chronological_split(series, args.holdout_len, args.train_len)
    train = split.train[["timestamp", *TARGETS]].reset_index(drop=True)

    folds = E.make_folds(len(train), horizon=args.holdout_len,
                         min_train=E.DEFAULT_MIN_TRAIN, step=E.DEFAULT_STEP)
    E.assert_folds_valid(folds, len(train))
    dev = E.dev_holdout_fold(len(train), args.holdout_len)

    candidates = {**B.BASELINES, **MO.candidate_models()}
    config = {"targets": {}, "selection": {}, "levels": list(LEVELS), "folds": [f.origin for f in folds]}
    interval_rows = []
    horizon_rows = []

    for target in TARGETS:
        values = train[target].to_numpy(dtype=float)
        results = {name: E.evaluate_candidate(cand, values, folds, target, name=name)
                   for name, cand in candidates.items()}
        table = E.results_table(results.values())
        winner, table = E.select_by_rule(table, COMPLEXITY_ORDER)
        res = results[winner]

        row = table.set_index("candidate").loc[winner]
        best_row = table.loc[table["rmse_mean_per_origin"].idxmin()]
        config["targets"][target] = {
            "model": winner,
            "rmse_mean_per_origin": float(row["rmse_mean_per_origin"]),
            "mae": float(row["mae"]),
            "mase": None if not np.isfinite(row["mase"]) else float(row["mase"]),
            "n_folds": int(row["n_folds"]),
        }
        config["selection"][target] = {
            "lowest_rmse_candidate": str(best_row["candidate"]),
            "lowest_rmse": float(best_row["rmse_mean_per_origin"]),
            "se_of_best": float(best_row["rmse_se_across_origins"])
            if np.isfinite(best_row["rmse_se_across_origins"]) else 0.0,
            "tied_candidates": table.loc[table["within_1se_of_best"], "candidate"].tolist(),
        }

        # Intervals: residual quantiles by horizon from the selected model's folds.
        q = I.quantiles_by_horizon(res.residuals, levels=LEVELS)
        for level, arr in q.items():
            for h, value in enumerate(np.asarray(arr, dtype=float), start=1):
                interval_rows.append({
                    "target": target, "level": level, "horizon_step": h,
                    "residual_quantile": float(value),
                })
        for level in (0.50, 0.80, 0.90, 0.95):
            cov = I.leave_one_origin_out_coverage(res.y_true, res.y_pred, level, levels=LEVELS)
            config["selection"][target].setdefault("loyo_coverage", {})[str(level)] = (
                None if not np.isfinite(cov) else float(cov)
            )
        for h, rmse_h in enumerate(res.per_horizon_rmse(), start=1):
            horizon_rows.append({"target": target, "horizon_step": h, "rmse": float(rmse_h)})

        # Dev-holdout fold: mirrors the final protocol once, on training data only.
        dev_res = E.evaluate_candidate(candidates[winner], values, [dev], target, name=winner)
        config["selection"][target]["dev_holdout"] = {
            "origin": dev.origin,
            "rmse": float(dev_res.per_origin_rmse()[0]),
            "mae": float(dev_res.per_origin_mae()[0]),
        }
        print(f"{target}: selected {winner} (lowest rmse {best_row['candidate']})")

    config["ridge_alpha"] = 10.0
    config["seed"] = 0
    config["note"] = ("Frozen output of scripts/select_model.py. The predict command refits the "
                      "named configuration on the training window of whatever data it is given.")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(config, indent=2))
    ART.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(interval_rows).to_csv(ART / "residual_quantiles.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(ART / "selected_per_horizon_rmse.csv", index=False)
    print(f"wrote {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
