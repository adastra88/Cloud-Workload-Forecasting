"""Split tests: sizes, chronological order, no overlap, graceful shrink."""

from __future__ import annotations

import pandas as pd
import pytest

from cwf.splits import chronological_split, describe, MIN_TRAIN_OBS
from conftest import DATA_DIR


def _series(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [i * 300 for i in range(n)],
        "active_vm_count": [10.0] * n,
        "aggregate_cpu_utilization": [40.0 + (i % 7) for i in range(n)],
    })


def test_split_sizes_110_24_on_public_data():
    path = DATA_DIR / "vm_cpu_readings.csv.gz"
    if not path.exists():
        pytest.skip("public release not present")
    from cwf.data import load_readings

    readings = load_readings(DATA_DIR)
    assert readings["timestamp"].nunique() == 134
    split = chronological_split(_series(134))
    assert len(split.train) == 110
    assert len(split.holdout) == 24
    assert split.train["timestamp"].max() < split.holdout["timestamp"].min()


def test_split_chronological_and_disjoint():
    split = chronological_split(_series(134))
    assert split.train["timestamp"].is_monotonic_increasing
    assert split.holdout["timestamp"].is_monotonic_increasing
    assert set(split.train["timestamp"]).isdisjoint(set(split.holdout["timestamp"]))
    assert split.holdout["timestamp"].iloc[-1] == 133 * 300


def test_split_holdout_is_last_n():
    split = chronological_split(_series(50), holdout_len=10, train_len=38)
    assert split.holdout["timestamp"].tolist() == [40 * 300 + i * 300 for i in range(10)]


def test_split_shrinks_train_window_when_short():
    split = chronological_split(_series(20), holdout_len=5, train_len=110)
    assert len(split.holdout) == 5
    assert len(split.train) == 15


def test_split_rejects_too_short_series():
    with pytest.raises(ValueError):
        chronological_split(_series(10), holdout_len=10)


def test_split_rejects_unsorted_and_duplicates():
    bad = _series(30).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        chronological_split(bad, holdout_len=5)
    dup = pd.concat([_series(30), _series(30).iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        chronological_split(dup, holdout_len=5)


def test_describe_reports_cadence():
    split = chronological_split(_series(134))
    info = describe(split)
    assert info["cadence"] == 300.0
    assert info["horizon_hours"] == pytest.approx(2.0)
    assert info["contiguous_grid"] is True


def test_min_train_obs_sanity():
    assert MIN_TRAIN_OBS >= 12
