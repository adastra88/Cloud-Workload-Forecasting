"""Target construction: ``active_vm_count`` and ``aggregate_cpu_utilization``.

Contract (DATASET.md §4)
------------------------
active_vm_count(t)          = #distinct vm_id with a valid CPU reading at t
aggregate_cpu_utilization(t)= mean of VM-level avg_cpu at t, percentage [0, 100]

Design choices (documented in docs/protocol.md and docs/DECISIONS.md):
* A reading is valid when ``avg_cpu`` is finite and in [0, 100]
  (:func:`cwf.data.valid_reading_mask`).
* Missing / invalid readings are *excluded* from the mean, never zero-filled.
  Zero-filling would understate demand and is the single most common way to
  bias a utilisation mean.
* Duplicate ``(vm_id, timestamp)`` rows are de-duplicated deterministically
  (mean of the duplicates) so a VM is counted at most once per timestamp.
* Gaps in the timestamp grid are NOT resampled at the modelling cadence: the
  series is indexed by the observed timestamps, and the cadence is inferred
  from the data (:func:`infer_step_seconds`).  For the public release the grid
  is complete, so this is a no-op; for a smoke dataset with gaps it means we
  never invent a target value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import valid_reading_mask

TARGETS = ("active_vm_count", "aggregate_cpu_utilization")


def infer_step_seconds(timestamps) -> float:
    """Modal spacing between distinct timestamps, in the raw timestamp unit.

    The challenge stores seconds; the function returns the modal difference so
    callers do not hard-code 300.  Falls back to 1.0 for a single timestamp.
    """
    ts = np.sort(np.unique(np.asarray(timestamps, dtype=float)))
    if ts.size < 2:
        return 1.0
    diffs = np.diff(ts)
    values, counts = np.unique(np.round(diffs, 6), return_counts=True)
    return float(values[int(np.argmax(counts))])


def _dedupe(readings: pd.DataFrame) -> pd.DataFrame:
    keys = ["vm_id", "timestamp"]
    if readings.duplicated(keys).any():
        return readings.groupby(keys, as_index=False).mean(numeric_only=True)
    return readings


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """Return the aggregate series indexed by ``timestamp`` ascending.

    Columns: ``timestamp``, ``active_vm_count``, ``aggregate_cpu_utilization``,
    plus ``n_valid`` (readings used) and ``active_vm_count_all_rows`` (VMs with
    any row at t, valid or not) which is useful diagnostics only.
    """
    frame = _dedupe(readings)
    valid = valid_reading_mask(frame)
    good = frame.loc[valid]

    if good.empty:
        counts = pd.Series(dtype=float)
        means = pd.Series(dtype=float)
    else:
        grouped = good.groupby("timestamp")["avg_cpu"]
        counts = grouped.size().astype(float)
        means = grouped.mean().astype(float)

    all_counts = frame.groupby("timestamp").size().astype(float)
    out = pd.DataFrame({"timestamp": np.sort(frame["timestamp"].unique())})
    out = out.set_index("timestamp")
    out["active_vm_count"] = counts.reindex(out.index)
    out["aggregate_cpu_utilization"] = means.reindex(out.index)
    out["n_valid"] = out["active_vm_count"]
    out["active_vm_count_all_rows"] = all_counts.reindex(out.index)
    # Timestamps with zero valid readings cannot contribute a mean: leave NaN
    # rather than inventing a value.  On the public release this never happens
    # (asserted by tests).
    out["active_vm_count"] = out["active_vm_count"].fillna(0.0)
    out = out.reset_index()
    return out


def require_complete(series: pd.DataFrame, name: str = "target series", context: str = "") -> None:
    """Raise a clear error when a target cannot be evaluated."""
    for col in TARGETS:
        if col not in series:
            raise ValueError(f"{name} has no column {col!r}")
        bad = series[col].isna()
        if bad.any():
            first = series.loc[bad, "timestamp"].iloc[0]
            raise ValueError(
                f"{name} has {int(bad.sum())} NaN values in {col!r} "
                f"(first at timestamp {first}).{(' Context: ' + context) if context else ''}"
            )


def clip_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Clip to the contract ranges (documented post-processing, not modelling)."""
    out = frame.copy()
    out["active_vm_count"] = out["active_vm_count"].clip(lower=0.0).round().astype(int)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0.0, 100.0)
    return out


# --------------------------------------------------------------------------- #
# optional extra targets (labelled "extras" everywhere they are used)
# --------------------------------------------------------------------------- #
EXTRAS = ("share_hot_vms", "cross_vm_p95_avg_cpu", "aggregate_max_cpu")


def build_extra_targets(readings: pd.DataFrame, hot_threshold: float) -> pd.DataFrame:
    """Extra operational targets.  ``hot_threshold`` must be fixed on training data.

    * ``share_hot_vms``          : fraction of active VMs with avg_cpu >= threshold
    * ``cross_vm_p95_avg_cpu``   : cross-VM 95th percentile of avg_cpu
    * ``aggregate_max_cpu``      : mean of VM-level max_cpu (tail-ish load per VM)
    """
    frame = _dedupe(readings)
    frame = frame.loc[valid_reading_mask(frame)]
    grouped = frame.groupby("timestamp")
    out = pd.DataFrame({
        "timestamp": np.sort(frame["timestamp"].unique()),
    })
    hot = (frame["avg_cpu"] >= hot_threshold).groupby(frame["timestamp"]).mean()
    p95 = grouped["avg_cpu"].quantile(0.95)
    agg_max = grouped["max_cpu"].mean()
    out = out.set_index("timestamp")
    out["share_hot_vms"] = hot.reindex(out.index)
    out["cross_vm_p95_avg_cpu"] = p95.reindex(out.index)
    out["aggregate_max_cpu"] = agg_max.reindex(out.index)
    return out.reset_index()
