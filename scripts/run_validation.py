"""Phase 3/4: rolling-origin evaluation of every candidate, training window only.

This script never reads a holdout value.  It builds the full target series only
because the split function is the single owner of the boundary (R1), then
restricts everything to ``split.train``.
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
from cwf import models as MO  # noqa: E402
from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import TARGETS, build_targets, infer_step_seconds  # noqa: E402

ART = ROOT / "reports" / "artifacts"
SEED = 0

COMPLEXITY_ORDER = [
    "persistence", "ma6", "ma12", "ma24", "drift", "seasonal_naive12",
    "ets_holt", "ets_damped", "ridge_direct", "gbm_direct",
]


def load_training_series(data_dir: Path, holdout_len: int, train_len: int | None):
    dataset = load_dataset(data_dir)
    series = build_targets(dataset.readings)
    step = infer_step_seconds(series["timestamp"])
    split = chronological_split(series, holdout_len=holdout_len, train_len=train_len)
    # R1: holdout values are dropped here and never used again in this script.
    train = split.train[["timestamp", *TARGETS]].reset_index(drop=True)
    return dataset, series, train, step


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--holdout-len", type=int, default=24)
    parser.add_argument("--train-len", type=int, default=110)
    parser.add_argument("--horizon", type=int, default=E.DEFAULT_HORIZON)
    parser.add_argument("--min-train", type=int, default=E.DEFAULT_MIN_TRAIN)
    parser.add_argument("--step", type=int, default=E.DEFAULT_STEP)
    parser.add_argument("--out", default=str(ART / "validation_results.csv"))
    args = parser.parse_args(argv)

    np.random.seed(SEED)
    ART.mkdir(parents=True, exist_ok=True)

    dataset, series, train, step = load_training_series(Path(args.data), args.holdout_len, args.train_len)
    print(f"training observations: {len(train)}  cadence: {step:.0f}s  "
          f"window: {train['timestamp'].iloc[0]:.0f}..{train['timestamp'].iloc[-1]:.0f}")

    folds = E.make_folds(len(train), horizon=args.horizon, min_train=args.min_train, step=args.step)
    E.assert_folds_valid(folds, len(train))
    pd.DataFrame([{"origin": f.origin, "horizon": f.horizon,
                   "train_len": f.origin + 1} for f in folds]).to_csv(
        ART / "validation_folds.csv", index=False)
    print(f"folds: {[f.origin for f in folds]}")

    candidates = {**B.BASELINES, **B.DIAGNOSTIC_BASELINES, **MO.candidate_models()}

    all_tables = []
    per_horizon_all = []
    per_origin_all = []
    selected = {}
    for target in TARGETS:
        values = train[target].to_numpy(dtype=float)
        results = []
        for name, cand in candidates.items():
            res = E.evaluate_candidate(cand, values, folds, target, name=name)
            results.append(res)
            if name in B.DIAGNOSTIC_BASELINES:
                continue
        table = E.results_table([r for r in results if r.name not in B.DIAGNOSTIC_BASELINES])
        winner, table = E.select_by_rule(table, COMPLEXITY_ORDER)
        diagnostics = E.results_table([r for r in results if r.name in B.DIAGNOSTIC_BASELINES])
        all_tables.append(table)
        if not diagnostics.empty:
            diag = diagnostics.copy()
            diag["selected"] = False
            diag["within_1se_of_best"] = False
            all_tables.append(diag)
        per_horizon_all.append(E.per_horizon_table(results).assign(target=target))
        per_origin_all.append(E.per_origin_table(results).assign(target=target))
        selected[target] = winner
        print(f"\n=== {target} ===")
        cols = ["candidate", "rmse_mean_per_origin", "rmse_se_across_origins", "mae", "mase", "selected"]
        print(table[cols].sort_values("rmse_mean_per_origin").to_string(index=False))
        print(f"  selected by pre-registered rule: {winner}")

    table_all = pd.concat(all_tables, ignore_index=True)
    table_all.to_csv(args.out, index=False)
    pd.concat(per_horizon_all, ignore_index=True).to_csv(ART / "validation_per_horizon.csv", index=False)
    pd.concat(per_origin_all, ignore_index=True).to_csv(ART / "validation_per_origin.csv", index=False)

    summary = {
        "n_train": int(len(train)),
        "step_seconds": float(step),
        "folds": [f.origin for f in folds],
        "horizon": args.horizon,
        "min_train": args.min_train,
        "step": args.step,
        "selected": selected,
        "seed": SEED,
    }
    (ART / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
