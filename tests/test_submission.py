"""Submission contract and basic metric sanity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwf import metrics as M

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions.csv"


def test_submitted_predictions_file_exists():
    assert PREDICTIONS.exists(), "run `python -m cwf.predict --data challenge_data --output predictions.csv`"


def test_submitted_predictions_have_exact_columns():
    frame = pd.read_csv(PREDICTIONS)
    assert list(frame.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def test_submitted_predictions_have_24_ordered_unique_rows():
    frame = pd.read_csv(PREDICTIONS)
    assert len(frame) == 24
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].is_unique


def test_submitted_predictions_match_holdout_timestamps():
    frame = pd.read_csv(PREDICTIONS)
    readings = pd.read_csv(ROOT / "challenge_data" / "vm_cpu_readings.csv.gz", usecols=["timestamp"])
    expected = np.sort(readings["timestamp"].unique())[-24:]
    np.testing.assert_array_equal(frame["timestamp"].to_numpy(), expected)


def test_submitted_predictions_are_finite_and_in_range():
    frame = pd.read_csv(PREDICTIONS)
    values = frame[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy()
    assert np.isfinite(values).all()
    assert (frame["active_vm_count"] >= 0).all()
    assert frame["aggregate_cpu_utilization"].between(0, 100).all()
    assert not frame.isna().any().any()


def test_validator_script_accepts_the_submission():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(PREDICTIONS)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def test_validator_rejects_a_bad_file(tmp_path):
    frame = pd.read_csv(PREDICTIONS).iloc[:-1]  # 23 rows
    bad_path = tmp_path / "short.csv"
    frame.to_csv(bad_path, index=False)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_submission.py"), str(bad_path)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode != 0, "validator must reject a 23-row submission"


def test_persistence_on_a_constant_series_has_zero_error():
    series = np.full(50, 7.0)
    forecast = np.full(10, series[-1])
    truth = series[-10:]
    assert M.rmse(truth, forecast) == 0.0
    assert M.mae(truth, forecast) == 0.0


def test_metrics_have_expected_values_on_a_toy_example():
    truth = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 2.0, 5.0])
    assert M.mae(truth, pred) == pytest.approx(1.0)
    assert M.rmse(truth, pred) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_mase_is_undefined_when_naive_scale_is_zero():
    truth = np.full(24, 5.0)
    pred = np.full(24, 5.0)
    naive_scale = 0.0
    value = M.mase(truth, pred, naive_scale)
    assert np.isnan(value)


def test_skill_score_zero_when_matching_the_baseline():
    mse = 5.0
    assert M.skill_score(mse, mse) == pytest.approx(0.0)


def test_skill_score_negative_when_worse_than_baseline():
    assert M.skill_score(10.0, 5.0) < 0


def test_skill_score_positive_when_better_than_baseline():
    assert M.skill_score(2.0, 5.0) > 0
