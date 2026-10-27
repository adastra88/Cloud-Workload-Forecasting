# Cloud workload forecasting

Forecast two aggregate signals 24 steps ahead (2 hours on a 5-minute grid) for a
10,000-VM slice of the Microsoft Azure Public Dataset V2:

| target | definition |
| --- | --- |
| `active_vm_count` | distinct VMs with a valid CPU reading at each timestamp |
| `aggregate_cpu_utilization` | mean VM-level `avg_cpu` at each timestamp, in [0, 100] |

The submission is `predictions.csv`: 24 rows, exactly
`timestamp,active_vm_count,aggregate_cpu_utilization`.

## Quick start

```bash
# 1. environment (Python 3.10+; 3.12 recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. data -> challenge_data/  (verifies the release checksums)
python scripts/download_data.py

# 3. the one command that produces the submission
python -m cwf.predict --data challenge_data --output predictions.csv

# 4. check it against the published contract
python scripts/validate_submission.py predictions.csv
```

`challenge_data/` is gitignored and never committed.

## Reproducing everything

`make` targets in order. Each is a thin wrapper around a script.

| target | produces |
| --- | --- |
| `make setup` | venv + editable install |
| `make data` | `challenge_data/` with checksums verified |
| `make targets` | target series, split, extra targets (`reports/artifacts/`) |
| `make eda` | training-window EDA artifacts and figures |
| `make baselines` / `make models` | rolling-origin validation tables |
| `make select` | `config/final_model.json` + `predictions.csv` |
| `make final` | one-shot holdout score (**run once**) |
| `make report` | capacity policy, post-hoc analysis |
| `make test` | pytest |

## What the pipeline does

1. **Load** both files, keep `avg_cpu` that is finite and in [0, 100].
2. **Build** the aggregate series at every timestamp in the file.
3. **Split** so the first 110 observations are train and the last 24 are
   holdout. One function owns this (`cwf.splits.chronological_split`). The
   predict path keeps the holdout *timestamps* and discards the holdout *values*
   immediately.
4. **Refit** the frozen configuration from `config/final_model.json` on the
   training window of whatever data is supplied.
5. **Forecast** 24 steps and map them onto the inferred holdout timestamps.
6. **Clip** to valid ranges and write three columns, no index.

Cadence, holdout length and timestamps are all inferred from the data. `24` is
only a CLI default, and if the training window is too short for the frozen
model the code falls back to the baseline and logs a warning instead of
crashing.

## Main modeling decisions

- **The baseline won, and the report says so.** Rolling-origin validation chose
  persistence for both targets. On the one-shot holdout, a 24-step moving
  average scored better on aggregate CPU (RMSE 0.1537 vs 0.1848). I followed the
  pre-registered rule anyway; retuning after seeing the holdout would have
  invalidated the exercise.
- `active_vm_count` is exactly constant over the supplied window, so it is
  forecast by persistence and scale-free metrics are reported as undefined.
  The code still handles varying counts (see `cwf/smoke.py` and the tests).
- No seasonal model: 9.2 h of training data cannot identify a daily or weekly
  cycle.
- Metadata lifetime statistics (`max_cpu`, `avg_cpu`, `p95_max_cpu`) and
  `timestamp_vm_deleted` are banned as features because they encode the future.
  See `docs/feature_audit.md`.

## Leakage controls

`docs/protocol.md` was committed before any model was evaluated. Holdout truth
is loaded in exactly one function, `scripts/final_score.py::load_holdout_truth`,
and `tests/test_leakage.py` parses the source tree to prove no other module
references it. The same file checks that randomizing or zeroing the holdout
readings leaves `predictions.csv` byte-identical.

## Layout

```
src/cwf/        data, targets, splits, features, baselines, models,
                evaluate, metrics, intervals, policy, predict, smoke
scripts/        download_data, run_eda, run_validation, select_model,
                validate_submission, final_score, posthoc_analysis,
                run_policy, evaluate_policy, run_extras
config/         final_model.json (frozen selection output)
notebooks/      thin walkthroughs that import from src
reports/        final_report.md, figures/, artifacts/
docs/           protocol, feature audit, decisions, journal, handoff note
tests/          pytest suite
```

## Documentation

- `reports/final_report.md` - the write-up.
- `docs/HANDOFF_SUMMARY.md` - plain-English summary of the whole effort.
- `docs/DECISIONS.md` - every non-trivial decision with alternatives.
- `docs/journal/` - short phase-by-phase notes.