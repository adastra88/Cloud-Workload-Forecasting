"""Phase 2 extras: hot-VM share, cross-VM P95 and aggregate max_cpu.

These are labelled *extras* everywhere: they are not part of the submitted
`predictions.csv` schema, but they sharpen the operational story ("the mean is
not the tail") and are used to explain why a mean-based provisioning rule
under-protects the fleet.

The hot threshold is fixed on **training-window readings only** so it can never
be tuned with holdout information.
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

from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split, train_readings  # noqa: E402
from cwf.targets import EXTRAS, build_extra_targets, build_targets  # noqa: E402

ART = ROOT / "reports" / "artifacts"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--holdout-len", type=int, default=24)
    parser.add_argument("--quantile", type=float, default=0.90,
                        help="percentile of training-window VM readings that defines a hot VM")
    args = parser.parse_args(argv)

    dataset = load_dataset(Path(args.data))
    series = build_targets(dataset.readings)
    split = chronological_split(series, args.holdout_len, None)
    train = split.train[["timestamp", "aggregate_cpu_utilization"]]
    tr_readings = train_readings(dataset.readings, split)

    positive = tr_readings.loc[tr_readings["avg_cpu"] > 0, "avg_cpu"].to_numpy(dtype=float)
    hot_threshold = float(np.quantile(positive, args.quantile))

    extras = build_extra_targets(dataset.readings, hot_threshold)
    extras_train = extras[extras["timestamp"] <= split.train_end_timestamp]

    rows = []
    for name in EXTRAS:
        values = extras_train[name].to_numpy(dtype=float)
        agg = train["aggregate_cpu_utilization"].to_numpy(dtype=float)
        rows.append({
            "extra_target": name,
            "n_train": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "corr_with_aggregate_cpu": float(np.corrcoef(agg, values)[0, 1]) if np.std(values) > 0 else float("nan"),
            "ratio_mean_to_aggregate": float(np.mean(values) / np.mean(agg)) if np.mean(agg) else float("nan"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(ART / "extra_targets_train.csv", index=False)
    (ART / "extra_targets_meta.json").write_text(json.dumps({
        "hot_threshold_pp": hot_threshold,
        "threshold_quantile": args.quantile,
        "threshold_source": "training-window readings only (all VMs, avg_cpu > 0)",
    }, indent=2))
    print(table.to_string(index=False))
    print(f"\nhot threshold (train P{int(args.quantile*100)}): {hot_threshold:.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
