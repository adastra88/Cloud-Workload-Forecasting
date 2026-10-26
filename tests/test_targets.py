"""Target construction tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwf.targets import build_targets, clip_targets, infer_step_seconds


def _readings(rows):
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_each_vm_counted_once_per_timestamp():
    frame = _readings([
        (0, "a", 1, 2, 10.0),
        (0, "a", 1, 2, 30.0),  # duplicate key -> de-duplicated, still one VM
        (0, "b", 1, 2, 50.0),
    ])
    out = build_targets(frame)
    assert out.loc[0, "active_vm_count"] == 2
    assert out.loc[0, "aggregate_cpu_utilization"] == pytest.approx((20.0 + 50.0) / 2)


def test_missing_readings_are_not_zero_filled():
    frame = _readings([
        (0, "a", 1, 2, 40.0),
        (0, "b", 1, 2, 60.0),
        (300, "a", 1, 2, 40.0),
    ])
    out = build_targets(frame)
    assert out.loc[1, "active_vm_count"] == 1
    assert out.loc[1, "aggregate_cpu_utilization"] == pytest.approx(40.0)  # not 20.0


def test_invalid_readings_excluded():
    frame = _readings([
        (0, "a", 1, 2, 40.0),
        (0, "b", 1, 2, 150.0),  # out of range
        (0, "c", np.nan, np.nan, np.nan),
    ])
    out = build_targets(frame)
    assert out.loc[0, "active_vm_count"] == 1
    assert out.loc[0, "aggregate_cpu_utilization"] == pytest.approx(40.0)


def test_aggregate_within_range_and_timestamp_sorted():
    rng = np.random.default_rng(0)
    rows = [(int(t), f"vm-{v}", 0, 100, float(rng.uniform(0, 100))) for t in range(0, 3000, 300) for v in range(20)]
    out = build_targets(_readings(rows))
    assert out["timestamp"].is_monotonic_increasing
    assert out["aggregate_cpu_utilization"].between(0, 100).all()
    assert (out["active_vm_count"] == 20).all()


def test_no_valid_readings_gives_nan_not_invented_value():
    frame = _readings([(0, "a", 1, 2, 140.0), (300, "a", 1, 2, 60.0)])
    out = build_targets(frame)
    assert np.isnan(out.loc[0, "aggregate_cpu_utilization"])
    assert not np.isnan(out.loc[1, "aggregate_cpu_utilization"])


def test_clip_targets_respects_contract():
    frame = pd.DataFrame({
        "timestamp": [0, 300],
        "active_vm_count": [-1.0, 2.4],
        "aggregate_cpu_utilization": [120.0, -3.0],
    })
    out = clip_targets(frame)
    assert out["active_vm_count"].tolist() == [0, 2]
    assert out["aggregate_cpu_utilization"].tolist() == [100.0, 0.0]
    assert out["active_vm_count"].dtype.kind == "i"


def test_infer_step_seconds():
    assert infer_step_seconds([0, 300, 600, 900]) == 300.0
    assert infer_step_seconds([5]) == 1.0
    assert infer_step_seconds([0, 300, 600, 900, 1500]) == 300.0  # modal, gap tolerated
