"""Validate a predictions.csv against the submission contract.

Checks (SUBMISSION.md):
* exactly the columns ``timestamp,active_vm_count,aggregate_cpu_utilization``
  with no index column;
* exactly ``--holdout-len`` rows (default 24);
* timestamps identical to the holdout timestamps derived from the release data,
  in the same raw representation, chronological, no duplicates;
* no missing values, everything finite;
* ``active_vm_count >= 0`` and integer-valued;
* ``aggregate_cpu_utilization`` in [0, 100].

Note: this validator uses the holdout *timestamps* from the data, which the
protocol allows (they are the output index), never the holdout values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwf.data import load_dataset  # noqa: E402
from cwf.predict import SUBMISSION_COLUMNS  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import build_targets  # noqa: E402


def validate(path: Path, data_dir: Path | None, holdout_len: int) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"{path} does not exist"]
    frame = pd.read_csv(path)
    header = path.read_text(encoding="utf-8").splitlines()[0].strip()
    if header != ",".join(SUBMISSION_COLUMNS):
        problems.append(f"header is {header!r}, expected {','.join(SUBMISSION_COLUMNS)!r}")
    if list(frame.columns) != SUBMISSION_COLUMNS:
        problems.append(f"columns are {list(frame.columns)}, expected {SUBMISSION_COLUMNS}")
        return problems
    if len(frame) != holdout_len:
        problems.append(f"row count is {len(frame)}, expected {holdout_len}")
    if frame.isna().any().any():
        problems.append(f"missing values in {frame.columns[frame.isna().any()].tolist()}")
    for col in SUBMISSION_COLUMNS[1:]:
        if not np.all(np.isfinite(frame[col].to_numpy(dtype=float))):
            problems.append(f"non-finite values in {col}")
    if frame["timestamp"].duplicated().any():
        problems.append("duplicate timestamps")
    if not frame["timestamp"].is_monotonic_increasing:
        problems.append("timestamps are not chronological")
    if (frame["active_vm_count"] < 0).any():
        problems.append("negative active_vm_count")
    if not np.allclose(frame["active_vm_count"], np.round(frame["active_vm_count"])):
        problems.append("active_vm_count is not integer-valued")
    cpu = frame["aggregate_cpu_utilization"].to_numpy(dtype=float)
    if ((cpu < 0) | (cpu > 100)).any():
        problems.append("aggregate_cpu_utilization outside [0, 100]")

    if data_dir is not None and Path(data_dir).exists():
        series = build_targets(load_dataset(Path(data_dir)).readings)
        expected = chronological_split(series, holdout_len=holdout_len, train_len=None)
        exp_ts = expected.holdout["timestamp"].tolist()
        got_ts = frame["timestamp"].tolist()
        if len(exp_ts) == len(got_ts):
            if not np.allclose(np.asarray(got_ts, dtype=float), np.asarray(exp_ts, dtype=float)):
                problems.append("timestamps do not match the holdout timestamps of the data")
        else:
            problems.append("timestamp count does not match the derived holdout length")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="predictions.csv")
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--holdout-len", type=int, default=24)
    args = parser.parse_args(argv)
    problems = validate(Path(args.path), Path(args.data) if args.data else None, args.holdout_len)
    if problems:
        print("INVALID:")
        for p in problems:
            print(f" - {p}")
        return 1
    print(f"OK: {args.path} satisfies the submission contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
