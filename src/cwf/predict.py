"""The submission command.

    python -m cwf.predict --data challenge_data --output predictions.csv

What it does, end to end:

1. Load ``vm_cpu_readings`` from ``--data`` (metadata is optional and is never a
   model feature — see ``docs/feature_audit.md``).
2. Build the full aggregate target series with ``cwf.targets.build_targets``.
3. Split once with ``cwf.splits.chronological_split``.  **The holdout values are
   dropped here.**  Only the ``holdout_len`` timestamps survive, as the output
   index (R1).  Consequently the output cannot depend on holdout readings: the
   holdout-corruption invariance test (R9) passes by construction.
4. Fit the model named in ``config/final_model.json`` on the training window of
   whatever data was supplied — it refits, it does not replay stored numbers, so
   a tiny smoke dataset produces a valid file from its own data.
5. Forecast ``holdout_len`` steps, clip to the contract ranges, and write
   exactly three columns.

If the training window is too short for the chosen model, or the recorded model
is unknown, the command logs a warning and falls back to persistence.  It always
writes a valid file rather than crashing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as B
from . import models as MO
from .data import load_dataset
from .splits import DEFAULT_HOLDOUT_LEN, MIN_TRAIN_OBS, chronological_split
from .targets import TARGETS, build_targets, clip_targets, infer_step_seconds

DEFAULT_CONFIG = Path("config/final_model.json")
SUBMISSION_COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


class _Registry:
    """Maps the frozen configuration names to fresh, unfitted estimators."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def get(self, name: str):
        if name in B.BASELINES:
            return B.BASELINES[name]
        if name in B.DIAGNOSTIC_BASELINES:
            return B.DIAGNOSTIC_BASELINES[name]
        models = MO.candidate_models(alpha=self.alpha)
        return models.get(name)


def load_config(path: Path | None) -> dict:
    """Read the frozen selection output; return an empty dict when absent."""
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def build_predictions(
    data_dir: Path,
    config: dict,
    holdout_len: int = DEFAULT_HOLDOUT_LEN,
    train_len: int | None = None,
    log=None,
) -> tuple[pd.DataFrame, dict]:
    """Produce the submission frame plus a diagnostics dict.

    The diagnostics deliberately contain no target values.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    dataset = load_dataset(data_dir)
    series = build_targets(dataset.readings)
    if series.empty:
        raise SystemExit(f"no readings found under {data_dir}; cannot forecast")

    if holdout_len >= len(series):
        new_holdout = max(1, len(series) // 4)
        log(f"warning: series has {len(series)} observations, shrinking holdout "
            f"from {holdout_len} to {new_holdout}")
        holdout_len = new_holdout

    step = infer_step_seconds(series["timestamp"])
    split = chronological_split(series, holdout_len=holdout_len, train_len=train_len)
    holdout_timestamps = split.holdout["timestamp"].to_numpy()
    train = split.train[["timestamp", *TARGETS]].reset_index(drop=True)

    n_train = len(train)
    diagnostics = {
        "data_dir": str(data_dir),
        "n_series_observations": int(len(series)),
        "n_training_observations": int(n_train),
        "holdout_len": int(holdout_len),
        "step_seconds": float(step),
        "n_vms": int(dataset.readings["vm_id"].nunique()),
        "models": {},
        "fallbacks": [],
    }

    forecasts = {}
    for target in TARGETS:
        chosen = (config.get("targets", {}).get(target, {}) or {}).get("model", "persistence")
        values = train[target].to_numpy(dtype=float)
        model = _Registry(alpha=float(config.get("ridge_alpha", 10.0))).get(chosen)
        if model is None:
            log(f"warning: unknown model {chosen!r} for {target}; falling back to persistence")
            diagnostics["fallbacks"].append({"target": target, "reason": f"unknown model {chosen!r}"})
            model, chosen = B.persistence, "persistence"
        if n_train < MIN_TRAIN_OBS and chosen not in B.BASELINES:
            log(f"warning: only {n_train} training observations; using persistence for {target}")
            diagnostics["fallbacks"].append({"target": target, "reason": "training window too short"})
            model, chosen = B.persistence, "persistence"

        fn = model.forecast if hasattr(model, "forecast") else model
        try:
            pred = np.asarray(fn(values, holdout_len), dtype=float).ravel()
        except Exception as exc:  # noqa: BLE001 - never fail to submit
            log(f"warning: {chosen} failed for {target} ({exc}); using persistence")
            diagnostics["fallbacks"].append({"target": target, "reason": f"{type(exc).__name__}"})
            pred = np.asarray(B.persistence(values, holdout_len), dtype=float)
        if pred.size != holdout_len:
            pred = np.resize(pred, holdout_len)
        if not np.all(np.isfinite(pred)):
            bad = int((~np.isfinite(pred)).sum())
            log(f"warning: {chosen} produced {bad} non-finite values for {target}; "
                f"replaced with the last training value")
            pred = np.where(np.isfinite(pred), pred, values[-1])
        forecasts[target] = pred
        diagnostics["models"][target] = {"model": chosen, "n_train": int(n_train)}

    out = pd.DataFrame({
        "timestamp": holdout_timestamps,
        "active_vm_count": forecasts["active_vm_count"],
        "aggregate_cpu_utilization": forecasts["aggregate_cpu_utilization"],
    })
    out = clip_targets(out)
    out = out[SUBMISSION_COLUMNS]
    return out, diagnostics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Produce predictions.csv from the release data.")
    parser.add_argument("--data", default="challenge_data", help="directory containing the release CSVs")
    parser.add_argument("--output", default="predictions.csv", help="output CSV path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="frozen model configuration")
    parser.add_argument("--holdout-len", type=int, default=DEFAULT_HOLDOUT_LEN)
    parser.add_argument("--train-len", type=int, default=None,
                        help="cap the training window length (default: all data before the holdout)")
    parser.add_argument("--diagnostics", default=None, help="optional JSON path for run diagnostics")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config) if args.config else None)
    predictions, diagnostics = build_predictions(
        Path(args.data), config, holdout_len=args.holdout_len, train_len=args.train_len,
    )
    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    if args.diagnostics:
        Path(args.diagnostics).write_text(json.dumps(diagnostics, indent=2))
    print(f"wrote {out_path} ({len(predictions)} rows, {len(predictions.columns)} columns)")
    for target, info in diagnostics["models"].items():
        print(f"  {target}: {info['model']} (trained on {info['n_train']} observations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
