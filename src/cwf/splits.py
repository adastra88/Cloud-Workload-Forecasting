"""The one and only train/holdout split.

R1 of the leakage protocol: the target series is built in full first, then this
module splits it.  In the prediction path the holdout *values* are dropped
immediately (see :func:`cwf.predict.build_predictions`); only the holdout
*timestamps* survive.

The split has two design parameters:

* ``holdout_len`` (default 24): number of final observations.
* ``train_len``   (default ``None``): if given, the training window is capped at
  that many observations *ending where the holdout starts* (public release: 110).
  If ``None`` the training window is "everything before the holdout".

Reviewers swapping in a tiny smoke dataset need neither number to match the
public release: the split adapts to whatever series it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_HOLDOUT_LEN = 24
DEFAULT_TRAIN_LEN = 110

#: Minimum observations the train path needs before it will attempt any model
#: beyond a trivial baseline.
MIN_TRAIN_OBS = 12


@dataclass(frozen=True)
class Split:
    """A chronological split of the aggregate target series."""

    train: pd.DataFrame
    holdout: pd.DataFrame

    @property
    def timestamps(self) -> list:
        return self.holdout["timestamp"].tolist()

    @property
    def train_end_timestamp(self):
        return self.train["timestamp"].iloc[-1]


def chronological_split(
    series: pd.DataFrame,
    holdout_len: int = DEFAULT_HOLDOUT_LEN,
    train_len: int | None = DEFAULT_TRAIN_LEN,
) -> Split:
    """Split ``series`` into (train, holdout); holdout = last ``holdout_len`` rows.

    ``series`` must have a ``timestamp`` column and be sorted ascending,
    unique.  When the series is too short for ``train_len + holdout_len`` the
    training window shrinks and the function still returns a usable split
    (callers decide whether they have enough history; see
    :attr:`Split.train` length and :data:`MIN_TRAIN_OBS`).
    """
    if holdout_len <= 0:
        raise ValueError("holdout_len must be positive")
    if holdout_len >= len(series):
        raise ValueError(
            f"series has {len(series)} observations, cannot carve out a "
            f"{holdout_len}-observation holdout"
        )
    if not series["timestamp"].is_monotonic_increasing:
        raise ValueError("series must be sorted by timestamp")
    if series["timestamp"].duplicated().any():
        raise ValueError("series contains duplicate timestamps")

    n = len(series)
    holdout_start = n - holdout_len
    if train_len is None:
        train_start = 0
    else:
        train_start = max(0, holdout_start - int(train_len))
    train = series.iloc[train_start:holdout_start].reset_index(drop=True)
    holdout = series.iloc[holdout_start:].reset_index(drop=True)
    return Split(train=train, holdout=holdout)


def train_readings(readings: pd.DataFrame, split: Split) -> pd.DataFrame:
    """VM-level readings belonging to the training window only.

    This is the guard for the holdout-corruption invariance test: anything
    downstream of this function cannot see a holdout reading, however the
    caller has mangled it.
    """
    return readings.loc[readings["timestamp"] <= split.train_end_timestamp].reset_index(drop=True)


def describe(split: Split, step_seconds: float | None = None) -> dict:
    """Structural description of a split (no target values)."""
    steps = split.holdout["timestamp"].diff().dropna()
    cadence = float(steps.mode().iloc[0]) if len(steps) else (step_seconds or float("nan"))
    return {
        "n_train": int(len(split.train)),
        "n_holdout": int(len(split.holdout)),
        "train_start": float(split.train["timestamp"].iloc[0]),
        "train_end": float(split.train["timestamp"].iloc[-1]),
        "holdout_start": float(split.holdout["timestamp"].iloc[0]),
        "holdout_end": float(split.holdout["timestamp"].iloc[-1]),
        "cadence": cadence,
        "horizon_hours": float(len(split.holdout) * cadence / 3600.0) if cadence else None,
        "contiguous_grid": bool(
            len(steps) == 0 or (abs(steps.to_numpy() - cadence) < 1e-9).all()
        ),
    }
