"""Frozen split contract: sizes, ordering, disjointness and fold geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwf.data import load_dataset
from cwf.evaluate import assert_folds_valid, make_folds
from cwf.splits import chronological_split
from cwf.targets import build_targets


@pytest.fixture(scope="module")
def public_series():
    return build_targets(load_dataset("challenge_data").readings)


def test_public_split_is_110_and_24(public_series):
    split = chronological_split(public_series, holdout_len=24, train_len=None)
    assert len(split.train) == 110
    assert len(split.holdout) == 24
    assert len(split.train) + len(split.holdout) == len(public_series) == 134


def test_split_is_chronological_and_disjoint(public_series):
    split = chronological_split(public_series, holdout_len=24, train_len=None)
    assert split.train["timestamp"].max() < split.holdout["timestamp"].min()
    assert set(split.train["timestamp"]).isdisjoint(set(split.holdout["timestamp"]))
    assert split.train["timestamp"].is_monotonic_increasing
    assert split.holdout["timestamp"].is_monotonic_increasing


def test_holdout_is_the_last_timestamps(public_series):
    split = chronological_split(public_series, holdout_len=24, train_len=None)
    expected = public_series["timestamp"].to_numpy()[-24:]
    np.testing.assert_array_equal(split.holdout["timestamp"].to_numpy(), expected)


def test_split_of_synthetic_length_uses_holdout_only():
    series = pd.DataFrame({
        "timestamp": np.arange(40) * 300,
        "active_vm_count": 3.0,
        "aggregate_cpu_utilization": 10.0,
    })
    split = chronological_split(series, holdout_len=8, train_len=None)
    assert len(split.holdout) == 8
    assert len(split.train) == 32


def test_folds_never_look_past_training_window(public_series):
    split = chronological_split(public_series, holdout_len=24, train_len=None)
    n = len(split.train)
    folds = make_folds(n, min_train=60, step=6, horizon=24)
    assert folds, "expected at least one fold"
    assert_folds_valid(folds, n)
    for fold in folds:
        assert fold.origin + fold.horizon <= n
        # the fold's own test window must also stay inside the training slice
        assert fold.test_slice.stop <= n


def test_fold_origins_match_protocol(public_series):
    split = chronological_split(public_series, holdout_len=24, train_len=None)
    folds = make_folds(len(split.train), min_train=60, step=6, horizon=24)
    assert [f.origin for f in folds] == [60, 66, 72, 78, 84]
    assert all(f.horizon == 24 for f in folds)
