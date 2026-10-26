"""One-shot holdout scoring.  **The only place in the project that loads holdout truth.**

Run exactly once, after ``predictions.csv`` and ``config/final_model.json`` are
frozen.  Whatever number comes out is reported; nothing is retuned afterwards
(R7).  Re-running it just re-scores the same frozen file, which is fine — what is
forbidden is changing the model in response.

Reports, for both targets and with every baseline scored identically:
MAE, RMSE, MASE (scale from the training window's in-sample naive error),
skill score versus the best baseline, relative improvement, and — for the
utilisation target — interval coverage and pinball loss on the band written by
the policy/interval step.
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
from cwf import intervals as I  # noqa: E402
from cwf import metrics as M  # noqa: E402
from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import TARGETS, build_targets  # noqa: E402

ART = ROOT / "reports" / "artifacts"
CONFIG = ROOT / "config" / "final_model.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--predictions", default="predictions.csv")
    parser.add_argument("--holdout-len", type=int, default=24)
    parser.add_argument("--out", default=str(ART / "final_scores.csv"))
    args = parser.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)

    # ---- the single read of holdout truth -------------------------------- #
    dataset = load_dataset(Path(args.data))
    series = build_targets(dataset.readings)
    split = chronological_split(series, holdout_len=args.holdout_len, train_len=None)
    truth = split.holdout[["timestamp", *TARGETS]].reset_index(drop=True)
    train = split.train[["timestamp", *TARGETS]].reset_index(drop=True)
    # --------------------------------------------------------------------- #

    preds = pd.read_csv(args.predictions)
    if not np.allclose(preds["timestamp"].to_numpy(dtype=float),
                       truth["timestamp"].to_numpy(dtype=float)):
        raise SystemExit("prediction timestamps do not line up with the holdout timestamps")

    config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    rows = []

    for target in TARGETS:
        y = truth[target].to_numpy(dtype=float)
        scale = M.naive_scale(train[target].to_numpy(dtype=float))

        model_pred = preds[target].to_numpy(dtype=float).reshape(1, -1)
        truth_2d = y.reshape(1, -1)

        baseline_preds = {}
        for name, fn in B.BASELINES.items():
            baseline_preds[name] = np.asarray(
                fn(train[target].to_numpy(dtype=float), len(y)), dtype=float
            ).reshape(1, -1)

        baseline_scores = {n: {"rmse": M.rmse(truth_2d, p), "mae": M.mae(truth_2d, p)}
                           for n, p in baseline_preds.items()}
        best_baseline = min(baseline_scores, key=lambda n: baseline_scores[n]["rmse"])

        model_rmse = M.rmse(truth_2d, model_pred)
        rows.append({
            "target": target,
            "candidate": config.get("targets", {}).get(target, {}).get("model", "persistence"),
            "rmse": model_rmse,
            "mae": M.mae(truth_2d, model_pred),
            "bias": M.bias(truth_2d, model_pred),
            "mase": M.mase(truth_2d, model_pred, scale),
            "scale_naive_mae_train": scale,
            "best_baseline": best_baseline,
            "best_baseline_rmse": baseline_scores[best_baseline]["rmse"],
            "skill_vs_best_baseline": M.skill_score(model_rmse, baseline_scores[best_baseline]["rmse"]),
            "rmse_improvement_vs_best_pct": 100.0 * (1 - model_rmse / baseline_scores[best_baseline]["rmse"])
            if baseline_scores[best_baseline]["rmse"] > 0 else float("nan"),
            "rmse_improvement_vs_persistence_pct": 100.0 * (1 - model_rmse / baseline_scores["persistence"]["rmse"])
            if baseline_scores["persistence"]["rmse"] > 0 else float("nan"),
        })
        for name, sc in baseline_scores.items():
            rows.append({
                "target": target, "candidate": f"baseline:{name}",
                "rmse": sc["rmse"], "mae": sc["mae"],
                "bias": M.bias(truth_2d, baseline_preds[name]),
                "mase": M.mase(truth_2d, baseline_preds[name], scale),
                "scale_naive_mae_train": scale, "best_baseline": best_baseline,
                "best_baseline_rmse": baseline_scores[best_baseline]["rmse"],
                "skill_vs_best_baseline": M.skill_score(sc["rmse"], baseline_scores[best_baseline]["rmse"]),
                "rmse_improvement_vs_best_pct": 100.0 * (1 - sc["rmse"] / baseline_scores[best_baseline]["rmse"])
                if baseline_scores[best_baseline]["rmse"] > 0 else float("nan"),
                "rmse_improvement_vs_persistence_pct": 100.0 * (1 - sc["rmse"] / baseline_scores["persistence"]["rmse"])
                if baseline_scores["persistence"]["rmse"] > 0 else float("nan"),
            })

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table[["target", "candidate", "rmse", "mae", "mase", "skill_vs_best_baseline"]]
          .to_string(index=False))
    print(f"\nwrote {args.out}")

    # Holdout truth summary for the report (allowed: R7 has now run).
    summary = truth[list(TARGETS)].describe().to_dict()
    (ART / "holdout_truth_summary.json").write_text(json.dumps(
        {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
