"""Synthetic smoke datasets.

Purpose
-------
Reviewers swap the public release for a tiny internal dataset to check the
pipeline is not hard-coded.  These builders let us run that check ourselves.
They intentionally create awkward inputs:

* few VMs (3) with **varying** active counts,
* missing readings (a VM disappears for a stretch),
* a truncated history (fewer timestamps than the defaults),
* a constant utilisation series,
* a non-contiguous timestamp grid.

Every builder returns a directory containing ``vm_cpu_readings.csv.gz`` (and,
when relevant, ``vm_metadata.csv.gz``) in exactly the release schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def _readings_from_matrix(values: np.ndarray, timestamps: np.ndarray, vm_ids: list[str]) -> pd.DataFrame:
    """``values[t, v]`` -> long readings frame; NaN entries are dropped (missing readings)."""
    rows = []
    for ti, ts in enumerate(timestamps):
        for vi, vm in enumerate(vm_ids):
            v = values[ti, vi]
            if np.isnan(v):
                continue
            rows.append((int(ts), vm, float(max(v - 5.0, 0.0)), float(min(v + 5.0, 100.0)), float(v)))
    return _frame(rows)


def make_basic(n_timestamps: int = 40, n_vms: int = 3, seed: int = SEED) -> pd.DataFrame:
    """Varying active counts, one gap in a VM's readings, wave-shaped usage."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n_timestamps) * 300
    values = np.zeros((n_timestamps, n_vms))
    for v in range(n_vms):
        base = 20.0 + 10.0 * v
        values[:, v] = base + 15.0 * np.sin(np.arange(n_timestamps) / 6.0 + v) + rng.normal(0, 2, n_timestamps)
    # VM 1 disappears for 8 steps in the middle -> active count varies 3 -> 2 -> 3.
    values[n_timestamps // 2 : n_timestamps // 2 + 8, 1] = np.nan
    values = np.clip(values, 0, 100)
    return _readings_from_matrix(values, ts, [f"vm-{i}" for i in range(n_vms)])


def make_short(seed: int = SEED) -> pd.DataFrame:
    """History far shorter than train_len + holdout_len (tests the fallback)."""
    return make_basic(n_timestamps=20, n_vms=2, seed=seed)


def make_constant(n_timestamps: int = 60, n_vms: int = 2) -> pd.DataFrame:
    """Every reading identical: active count constant, utilisation constant."""
    ts = np.arange(n_timestamps) * 300
    values = np.full((n_timestamps, n_vms), 42.0)
    return _readings_from_matrix(values, ts, [f"vm-{i}" for i in range(n_vms)])


def make_noncontiguous(seed: int = SEED) -> pd.DataFrame:
    """A hole in the timestamp grid (two steps missing)."""
    rng = np.random.default_rng(seed)
    ts = np.arange(50) * 300
    keep = np.ones(len(ts), dtype=bool)
    keep[20:22] = False  # timestamps 20*300 and 21*300 are absent entirely
    ts = ts[keep]
    n = len(ts)
    values = np.stack([30 + rng.normal(0, 3, n), 55 + rng.normal(0, 3, n)], axis=1)
    return _readings_from_matrix(np.clip(values, 0, 100), ts, ["vm-a", "vm-b"])


def make_missing_metadata(n_timestamps: int = 40, n_vms: int = 3) -> pd.DataFrame:
    """Basic dataset; caller writes no metadata file (metadata is optional)."""
    return make_basic(n_timestamps, n_vms)


def write_dataset(frame: pd.DataFrame, out_dir: str | Path, with_metadata: bool = False) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    if with_metadata:
        vms = sorted(frame["vm_id"].unique())
        meta = pd.DataFrame({
            "vm_id": vms,
            "subscription_id": [f"sub-{i % 2}" for i in range(len(vms))],
            "deployment_id": [f"dep-{i}" for i in range(len(vms))],
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": 10**9,
            "max_cpu": 90.0,
            "avg_cpu": 40.0,
            "p95_max_cpu": 80.0,
            "vm_category": "Interactive",
            "virtual_core_bucket": 2,
            "memory_gb_bucket": 4,
        })
        meta.to_csv(out / "vm_metadata.csv.gz", index=False, compression="gzip")
    return out


BUILDERS = {
    "basic": make_basic,
    "short": make_short,
    "constant": make_constant,
    "noncontiguous": make_noncontiguous,
}


def write_all(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    out = {}
    for name, builder in BUILDERS.items():
        out[name] = write_dataset(builder(), root / name, with_metadata=(name == "basic"))
    return out
