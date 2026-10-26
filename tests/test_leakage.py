"""Leakage guards: who may load holdout truth, and is the forecast actually blind to it?

These are the machine-checkable half of the R1-R9 protocol in ``docs/protocol.md``.
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

# The ONLY file allowed to read holdout target values.
HOLDOUT_READER = SCRIPTS / "final_score.py"
HOLDOUT_LOADER_NAMES = {"load_holdout_truth", "holdout_truth", "load_truth"}


def _iter_python_files():
    for base in (SRC, SCRIPTS):
        yield from base.rglob("*.py")


def test_only_final_score_loads_holdout_truth():
    """AST check: no other module may import the holdout loader."""
    offenders = []
    for path in _iter_python_files():
        if path == HOLDOUT_READER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
                if names & HOLDOUT_LOADER_NAMES:
                    offenders.append(f"{path.relative_to(ROOT)} imports {names & HOLDOUT_LOADER_NAMES}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in HOLDOUT_LOADER_NAMES:
                        offenders.append(f"{path.relative_to(ROOT)} imports {alias.name}")
            elif isinstance(node, ast.Attribute) and node.attr in HOLDOUT_LOADER_NAMES:
                offenders.append(f"{path.relative_to(ROOT)} references {node.attr}")
    assert not offenders, "holdout truth may only be loaded by scripts/final_score.py:\n" + "\n".join(offenders)


def test_final_score_defines_the_holdout_loader():
    source = HOLDOUT_READER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(HOLDOUT_READER))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert names & HOLDOUT_LOADER_NAMES, f"final_score.py should define one of {HOLDOUT_LOADER_NAMES}"


def test_predict_module_never_reads_holdout_values():
    """The prediction path keeps holdout timestamps, never holdout values."""
    source = (SRC / "cwf" / "predict.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "split" in source.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in HOLDOUT_LOADER_NAMES:
            pytest.fail("predict.py must not reference the holdout loader")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_predict(data_dir: Path, out: Path) -> str:
    cmd = [
        sys.executable, "-m", "cwf.predict",
        "--data", str(data_dir),
        "--output", str(out),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout


@pytest.fixture(scope="module")
def corruption_workspace(tmp_path_factory):
    """Three copies of the release: untouched, randomized holdout, zeroed holdout."""
    if not (ROOT / "challenge_data" / "vm_cpu_readings.csv.gz").exists():
        pytest.skip("public release not present")
    base = tmp_path_factory.mktemp("corruption")
    readings = pd.read_csv(ROOT / "challenge_data" / "vm_cpu_readings.csv.gz")
    timestamps = np.sort(readings["timestamp"].unique())
    holdout_ts = set(timestamps[-24:].tolist())
    is_holdout = readings["timestamp"].isin(holdout_ts)

    copies = {}
    variants = {
        "clean": readings.copy(),
        "randomized": readings.copy(),
        "zeroed": readings.copy(),
    }
    rng = np.random.default_rng(7)
    n_bad = int(is_holdout.sum())
    variants["randomized"].loc[is_holdout, "avg_cpu"] = rng.uniform(0, 100, n_bad)
    variants["zeroed"].loc[is_holdout, "avg_cpu"] = 0.0

    for name, frame in variants.items():
        d = base / name
        d.mkdir()
        frame.to_csv(d / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        meta = ROOT / "challenge_data" / "vm_metadata.csv.gz"
        if meta.exists():
            shutil.copy(meta, d / "vm_metadata.csv.gz")
        copies[name] = d
    return copies


def test_holdout_corruption_does_not_change_predictions(corruption_workspace, tmp_path):
    outputs = {}
    for name, data_dir in corruption_workspace.items():
        out = tmp_path / f"{name}.csv"
        _run_predict(data_dir, out)
        outputs[name] = out

    clean = _sha(outputs["clean"])
    assert _sha(outputs["randomized"]) == clean, "randomized holdout readings changed the forecast"
    assert _sha(outputs["zeroed"]) == clean, "zeroed holdout readings changed the forecast"


def test_corrupted_runs_match_submitted_predictions(corruption_workspace, tmp_path):
    if not (ROOT / "predictions.csv").exists():
        pytest.skip("predictions.csv not built")
    submitted = _sha(ROOT / "predictions.csv")
    out = tmp_path / "randomized.csv"
    _run_predict(corruption_workspace["randomized"], out)
    assert _sha(out) == submitted


def test_two_runs_are_byte_identical(tmp_path):
    if not (ROOT / "challenge_data" / "vm_cpu_readings.csv.gz").exists():
        pytest.skip("public release not present")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _run_predict(ROOT / "challenge_data", a)
    _run_predict(ROOT / "challenge_data", b)
    assert _sha(a) == _sha(b)
