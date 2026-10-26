"""Data loading and structural validation.

The *only* module that knows the on-disk schema of the ``data-v1`` release.
Nothing here looks at target values; it only enforces structure.

Leakage note
------------
``load_readings`` returns every row of the release. Leakage discipline is
enforced downstream: :mod:`cwf.splits` owns the train/holdout boundary and
:meth:`cwf.splits.Split.train_readings` never returns holdout rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"

READINGS_SCHEMA = {
    "timestamp": ("number",),
    "vm_id": ("string", "object"),
    "min_cpu": ("number",),
    "max_cpu": ("number",),
    "avg_cpu": ("number",),
}

METADATA_REQUIRED = ("vm_id",)
METADATA_STATIC = ("vm_category", "virtual_core_bucket", "memory_gb_bucket")


class DataContractError(ValueError):
    """Raised when the release does not satisfy the documented contract."""


@dataclass(frozen=True)
class Dataset:
    """Readings plus (optionally) metadata and a structural report."""

    readings: pd.DataFrame
    metadata: pd.DataFrame | None
    report: dict


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve(data_dir: str | os.PathLike) -> Path:
    path = Path(data_dir)
    if path.is_file():
        path = path.parent
    return path


def _normalise_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "string"


def validate_readings_schema(frame: pd.DataFrame) -> None:
    """Fail fast with an action-oriented message when columns are wrong."""
    missing = [c for c in READINGS_SCHEMA if c not in frame.columns]
    if missing:
        raise DataContractError(
            f"{READINGS_FILE} is missing required columns {missing}; "
            f"found {list(frame.columns)}"
        )
    for col, allowed in READINGS_SCHEMA.items():
        kind = _normalise_dtype(frame[col])
        if kind not in allowed:
            raise DataContractError(
                f"column {col!r} has dtype kind {kind!r}, expected one of {allowed}"
            )


def valid_reading_mask(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of readings usable as evidence of activity.

    A reading is "valid" when ``avg_cpu`` is finite and inside [0, 100].
    Documented in ``docs/protocol.md``; kept deliberately strict so that
    corrupted rows (see the holdout-corruption test) count as invalid.
    """
    avg = pd.to_numeric(frame["avg_cpu"], errors="coerce")
    return np.isfinite(avg) & avg.between(0.0, 100.0)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_readings(data_dir: str | os.PathLike) -> pd.DataFrame:
    """Load the VM CPU readings, sorted by ``(timestamp, vm_id)``.

    Raises :class:`FileNotFoundError` with a precise instruction when the
    release has not been extracted, as required by the working protocol.
    """
    path = _resolve(data_dir) / READINGS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the fixed `data-v1` GitHub release "
            "asset (data-v1.tar.gz) from "
            "https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1 "
            f"and extract it so that {READINGS_FILE} sits directly inside "
            f"{path.parent}."
        )
    frame = pd.read_csv(path)
    validate_readings_schema(frame)
    frame["vm_id"] = frame["vm_id"].astype(str)
    return frame.sort_values(["timestamp", "vm_id"], kind="mergesort").reset_index(drop=True)


def load_metadata(data_dir: str | os.PathLike) -> pd.DataFrame | None:
    """Load metadata, or return ``None`` when absent (metadata is optional)."""
    path = _resolve(data_dir) / METADATA_FILE
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    missing = [c for c in METADATA_REQUIRED if c not in frame.columns]
    if missing:
        raise DataContractError(f"{METADATA_FILE} is missing required columns {missing}")
    frame["vm_id"] = frame["vm_id"].astype(str)
    return frame.drop_duplicates("vm_id").reset_index(drop=True)


def load_dataset(data_dir: str | os.PathLike) -> Dataset:
    readings = load_readings(data_dir)
    metadata = load_metadata(data_dir)
    return Dataset(readings=readings, metadata=metadata, report=structural_report(readings, metadata))


# --------------------------------------------------------------------------- #
# structural report (schema-level; never emits target values)
# --------------------------------------------------------------------------- #
def structural_report(readings: pd.DataFrame, metadata: pd.DataFrame | None = None) -> dict:
    """Integrity facts computed over the *whole* file.

    Only structural quantities are returned: counts, grid properties,
    duplicate keys and invalid-reading counts. No per-timestamp target
    statistic is included, so this is safe to print before the one-shot
    holdout scoring.
    """
    timestamps = np.sort(readings["timestamp"].unique())
    per_vm = readings.groupby("vm_id").size()
    steps = np.diff(timestamps)
    uniq_steps = pd.Series(steps).value_counts().sort_index() if len(steps) else pd.Series(dtype=int)
    meta_vms = set(metadata["vm_id"]) if metadata is not None else None
    read_vms = set(readings["vm_id"])
    return {
        "n_rows": int(len(readings)),
        "n_vms": int(readings["vm_id"].nunique()),
        "n_timestamps": int(len(timestamps)),
        "timestamp_first": int(timestamps[0]) if len(timestamps) else None,
        "timestamp_last": int(timestamps[-1]) if len(timestamps) else None,
        "step_counts": {float(k): int(v) for k, v in uniq_steps.items()},
        "modal_step": float(uniq_steps.idxmax()) if len(uniq_steps) else None,
        "gap_count": int((np.asarray(list(uniq_steps.index)) > (uniq_steps.idxmax() if len(uniq_steps) else np.inf)).sum()) if len(uniq_steps) else 0,
        "n_gaps_vs_modal": int(sum(v for k, v in uniq_steps.items() if k != (uniq_steps.idxmax() if len(uniq_steps) else 0))),
        "readings_per_vm_min": int(per_vm.min()) if len(per_vm) else 0,
        "readings_per_vm_max": int(per_vm.max()) if len(per_vm) else 0,
        "readings_per_vm_modes": {int(k): int(v) for k, v in per_vm.value_counts().items()},
        "duplicate_vm_timestamp": int(readings.duplicated(["vm_id", "timestamp"]).sum()),
        "null_counts": {k: int(v) for k, v in readings.isna().sum().items()},
        "invalid_reading_rows": int((~valid_reading_mask(readings)).sum()),
        "min_gt_max_rows": int((readings["min_cpu"] > readings["max_cpu"]).sum()),
        "timestamp_monotonic_input": None,
        "metadata_rows": int(len(metadata)) if metadata is not None else 0,
        "metadata_vms_missing_readings": int(len(meta_vms - read_vms)) if meta_vms is not None else 0,
        "reading_vms_missing_metadata": int(len(read_vms - meta_vms)) if meta_vms is not None else 0,
    }


def write_sha256_verification(data_dir: str | os.PathLike, out_path: str | os.PathLike) -> pd.DataFrame:
    """Verify ``SHA256SUMS`` if present and record the outcome as a table."""
    import hashlib

    base = _resolve(data_dir)
    sums = base / "SHA256SUMS"
    rows = []
    if sums.exists():
        for line in sums.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            target = base / name
            if not target.exists():
                rows.append({"file": name, "expected": digest, "actual": None, "ok": False})
                continue
            h = hashlib.sha256()
            with open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            rows.append({"file": name, "expected": digest, "actual": actual, "ok": actual == digest})
    out = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
