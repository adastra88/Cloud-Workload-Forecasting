"""Data-loading contract tests (schema errors produce actionable messages)."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from cwf.data import DataContractError, load_metadata, load_readings, load_dataset, structural_report
from cwf.smoke import make_basic, write_dataset


def test_missing_data_dir_raises_helpful_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as err:
        load_readings(tmp_path / "nope")
    message = str(err.value)
    assert "data-v1" in message and "vm_cpu_readings.csv.gz" in message


def test_schema_error_names_missing_columns(tmp_path: Path):
    bad = make_basic().drop(columns=["avg_cpu"])
    write_dataset(bad, tmp_path)
    with pytest.raises(DataContractError) as err:
        load_readings(tmp_path)
    assert "avg_cpu" in str(err.value)


def test_metadata_optional(tmp_path: Path):
    write_dataset(make_basic(), tmp_path)
    dataset = load_dataset(tmp_path)
    assert dataset.metadata is None
    assert dataset.report["n_vms"] == 3


def test_metadata_loaded_when_present(tmp_path: Path):
    write_dataset(make_basic(), tmp_path, with_metadata=True)
    meta = load_metadata(tmp_path)
    assert meta is not None and len(meta) == 3


def test_structural_report_has_no_target_values():
    frame = make_basic()
    report = structural_report(frame)
    assert report["n_rows"] == len(frame)
    assert "aggregate_cpu_utilization" not in report
    assert all(not isinstance(v, pd.Series) for v in report.values())
