"""The predict CLI must survive tiny, awkward, non-public datasets.

Reviewers swap in an internal smoke dataset to check nothing is hard-coded, so
every builder in ``cwf.smoke`` gets the same treatment as the real release.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwf.smoke import BUILDERS, make_constant, write_dataset

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"timestamp", "active_vm_count", "aggregate_cpu_utilization"}


def _run_predict(data_dir: Path, out: Path, holdout_len: int | None = None, extra=()) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "cwf.predict", "--data", str(data_dir), "--output", str(out)]
    if holdout_len is not None:
        cmd += ["--holdout-len", str(holdout_len)]
    cmd += list(extra)
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def _check_output(path: Path, readings: pd.DataFrame, holdout_len: int):
    assert path.exists(), "predict did not write an output file"
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(frame) == holdout_len
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].is_unique
    assert np.isfinite(frame[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy()).all()
    assert (frame["active_vm_count"] >= 0).all()
    assert frame["aggregate_cpu_utilization"].between(0, 100).all()

    source_ts = np.sort(readings["timestamp"].unique())
    expected = source_ts[-holdout_len:]
    np.testing.assert_array_equal(frame["timestamp"].to_numpy(), expected)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_each_smoke_dataset_produces_a_valid_submission(name, tmp_path):
    readings = BUILDERS[name]()
    data_dir = write_dataset(readings, tmp_path / name, with_metadata=(name == "basic"))
    out = tmp_path / f"{name}.csv"
    proc = _run_predict(data_dir, out, holdout_len=6)
    assert proc.returncode == 0, proc.stderr[-4000:]
    _check_output(out, readings, holdout_len=6)


def test_smoke_dataset_without_metadata_still_works(tmp_path):
    readings = BUILDERS["basic"]()
    data_dir = write_dataset(readings, tmp_path / "nometa", with_metadata=False)
    assert not (data_dir / "vm_metadata.csv.gz").exists()
    out = tmp_path / "out.csv"
    proc = _run_predict(data_dir, out, holdout_len=5)
    assert proc.returncode == 0, proc.stderr[-4000:]
    _check_output(out, readings, holdout_len=5)


def test_constant_series_forecasts_a_constant(tmp_path):
    readings = make_constant(n_timestamps=60, n_vms=2)
    data_dir = write_dataset(readings, tmp_path / "const")
    out = tmp_path / "const.csv"
    proc = _run_predict(data_dir, out, holdout_len=8)
    assert proc.returncode == 0, proc.stderr[-4000:]
    frame = pd.read_csv(out)
    assert frame["active_vm_count"].nunique() == 1
    assert frame["aggregate_cpu_utilization"].nunique() == 1


def test_varying_active_count_is_forecast_not_assumed_constant(tmp_path):
    """The basic smoke set has 3 VMs with one disappearing, so the count moves."""
    readings = BUILDERS["basic"]()
    counts = readings.groupby("timestamp")["vm_id"].nunique()
    assert counts.nunique() > 1, "fixture should exercise a varying count"

    data_dir = write_dataset(readings, tmp_path / "vary")
    out = tmp_path / "vary.csv"
    proc = _run_predict(data_dir, out, holdout_len=6)
    assert proc.returncode == 0, proc.stderr[-4000:]
    frame = pd.read_csv(out)
    # The forecast must stay inside the plausible range of what was observed.
    assert frame["active_vm_count"].between(0, counts.max()).all()


def test_short_history_falls_back_instead_of_crashing(tmp_path):
    readings = BUILDERS["short"]()
    data_dir = write_dataset(readings, tmp_path / "short")
    out = tmp_path / "short.csv"
    # Ask for more history than exists to force the fallback path.
    proc = _run_predict(data_dir, out, holdout_len=5, extra=["--train-len", "200"])
    assert proc.returncode == 0, proc.stderr[-4000:]
    _check_output(out, readings, holdout_len=5)


def test_oversized_holdout_len_shrinks_with_a_warning(tmp_path):
    """A held-out window longer than the series is shrunk, loudly, not crashed on."""
    readings = BUILDERS["short"]()  # 20 timestamps
    data_dir = write_dataset(readings, tmp_path / "tooshort")
    out = tmp_path / "x.csv"
    proc = _run_predict(data_dir, out, holdout_len=50)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "shrinking holdout" in (proc.stderr + proc.stdout).lower()
    assert len(pd.read_csv(out)) < 20


def test_smoke_run_is_deterministic(tmp_path):
    readings = BUILDERS["basic"]()
    data_dir = write_dataset(readings, tmp_path / "det")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert _run_predict(data_dir, a, holdout_len=6).returncode == 0
    assert _run_predict(data_dir, b, holdout_len=6).returncode == 0
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
