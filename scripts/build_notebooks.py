"""Build and (optionally) execute the thin notebooks.

The notebooks are deliberately thin: they import from ``src/cwf`` and read the
committed artifacts, so they cannot drift from the pipeline.  Regenerate with
``python scripts/build_notebooks.py --execute``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
pd.set_option("display.width", 120)
ART = ROOT / "reports" / "artifacts"
print("project root:", ROOT)
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook_01():
    cells = [
        md("""
# 01 - What the data looks like

**Scope discipline.** Everything that looks for *patterns* here uses only the 110
training timestamps. The 24 holdout observations are never inspected in this
notebook: that is the point of `docs/protocol.md`. Structural checks (schema,
grid, duplicates) are allowed over the whole file because they do not reveal
target values.
"""),
        code(SETUP),
        md("""
## Structural facts

Read the readings file and check the grid, duplicates and value ranges. These are
integrity questions, not forecasting questions, so the full file is fair game.
"""),
        code("""
readings = pd.read_csv(ROOT / "challenge_data" / "vm_cpu_readings.csv.gz")
print("rows:", len(readings))
print("vms :", readings.vm_id.nunique())
ts = np.sort(readings.timestamp.unique())
print("timestamps:", len(ts))
print("gaps (s):", np.unique(np.diff(ts)))
print("duplicate (vm,timestamp) keys:", readings.duplicated(["vm_id", "timestamp"]).sum())
print("avg_cpu outside [0,100]:", (~readings.avg_cpu.between(0, 100)).sum())
"""),
        md("""
## The training window only

`cwf.splits.chronological_split` is the single owner of the split. From here on
we look at `split.train` and never at `split.holdout`.
"""),
        code("""
from cwf.data import load_dataset
from cwf.targets import build_targets
from cwf.splits import chronological_split

series = build_targets(load_dataset(ROOT / "challenge_data").readings)
split = chronological_split(series, holdout_len=24, train_len=None)
train = split.train
print("train:", len(train), "holdout:", len(split.holdout))
print("holdout timestamps are known, their values are not touched")

train[["timestamp", "active_vm_count", "aggregate_cpu_utilization"]].describe().T
"""),
        md("""
**Read this carefully.** `active_vm_count` has std 0: it is exactly 10,000 at all
110 training timestamps. That is a property of a complete rectangular release
slice, not of Azure. It means the count target is trivially persistent, and it
means every scale-free metric for that target divides by zero. We report that
honestly rather than manufacture complexity.
"""),
        code("""
print("distinct active counts in training:", train.active_vm_count.nunique())
print("aggregate CPU summary:")
print(train.aggregate_cpu_utilization.describe())
"""),
        md("""
## Is there trend or seasonality?

Two unit-root tests plus an autocorrelation function. With 9.2 hours we cannot
see a daily cycle, so we are asking a narrower question: is there a drift we can
exploit, and how fast does information decay?
"""),
        code("""
from cwf.data import load_dataset as _ld
import numpy as _np

acf = pd.read_csv(ART / "acf_train.csv")
print(acf.head(26).to_string(index=False))
"""),
        code("""
print(pd.read_csv(ART / "stationarity_trend.csv").to_string(index=False))
print()
print(pd.read_csv(ART / "acf_decay.csv").to_string(index=False))
"""),
        md("""
Autocorrelation is still 0.66 at lag 1 but drops inside the 95% band by lag 11
(about 55 minutes). ADF says "unit root not rejected", KPSS says "stationary not
rejected" at the 10% level: the series is close to the boundary, i.e. very
persistent but not obviously trending. Practically: last value and short averages
are both plausible, and neither will be much better than the other.
"""),
        md("""
## The mean hides the tail

The aggregate is about 9.4 percentage points. Individual machines reach 99.
Capacity planning on the mean alone would be a mistake.
"""),
        code("""
print(pd.read_csv(ART / "mean_hides_tails.csv").to_string(index=False))
print()
print(pd.read_csv(ART / "load_concentration.csv").to_string(index=False))
"""),
        md("""
## Composition versus load

Is aggregate movement caused by which VMs are active, or by the same VMs changing
their usage? Decompose the variance: between-VM level, common time effect, and
idiosyncratic residual.
"""),
        code("""
print(pd.read_csv(ART / "movement_decomposition.csv").to_string(index=False))
print()
print(pd.read_csv(ART / "composition_vs_load.csv").to_string(index=False))
"""),
        md("""
83.6% of the variance is simply *which machine you are looking at*. The
common-time component is 0.018%, so there is almost no "the whole fleet got
busier" signal to forecast. That is the core reason the aggregate is so hard to
beat with anything but a naive forecast.

## Operational takeaway

1. Forecast the count trivially - it does not move in this slice.
2. Forecast the aggregate with a strong baseline and honest uncertainty, not with
   a complex model on 110 points.
3. Tell the capacity team about the tail, because the mean is a poor proxy for
   the machine that matters.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells)


def notebook_02():
    cells = [
        md("""
# 02 - Validating like the holdout is real

The holdout values sit in the same file as the training data. Any evaluation that
touches them is worthless. So we build the evaluation entirely inside the 110
training points.
"""),
        code(SETUP),
        md("""
## The folds

Expanding window, horizon 24, origins every 6 steps. The constraint is
`origin + horizon <= 110` for every fold.
"""),
        code("""
from cwf.evaluate import make_folds, assert_folds_valid

folds = make_folds(110, min_train=60, step=6, horizon=24)
assert_folds_valid(folds, 110)
for f in folds:
    print(f"origin={f.origin:3d}  train={f.train_slice.stop:3d}  test={f.test_slice.start}-{f.test_slice.stop - 1}")
"""),
        md("""
Notice the last fold: origin 84, test window 85-108, still inside 110. Nothing
ever reaches into the holdout.

## Results as measured, not as hoped

The table below is the committed artifact from `scripts/run_validation.py`.
"""),
        code("""
results = pd.read_csv(ART / "validation_results.csv")
results.sort_values(["target", "rmse"]).to_string(index=False)
"""),
        md("""
Selection rule, frozen in `docs/protocol.md` before any of this ran: lowest
rolling-origin RMSE, and if a candidate is within one standard error of the best,
prefer the simpler. Applied to the table above it selects **persistence** for both
targets.

For `active_vm_count` every candidate has RMSE 0.0 because the series does not
move. MASE is undefined (zero naive scale). Both facts are printed, not hidden.

## How noisy is this?

Five overlapping origins is not a lot of evidence. Look at the spread across
origins before believing any ranking.
"""),
        code("""
per_origin = pd.read_csv(ART / "validation_per_origin.csv")
piv = per_origin[per_origin.target == "aggregate_cpu_utilization"].pivot_table(
    index="origin", columns="candidate", values="rmse")
piv[["persistence", "ma24", "drift", "ridge_direct", "gbm_direct"]].round(4)
"""),
        code("""
print(json.dumps(json.loads((ART / "validation_summary.json").read_text()), indent=2)[:2000])
"""),
        md("""
The ranking of persistence, drift and the GBM changes from origin to origin. With
standard error around 0.04 on an RMSE of 0.21, differences below about 0.04 are
noise. That is exactly why the rule broke ties toward simplicity - and exactly why
the holdout result later surprised us.

## Per-horizon behaviour

Errors usually grow with horizon. If they do not, the series is nearly a
random walk and persistence is the right answer.
"""),
        code("""
ph = pd.read_csv(ART / "validation_per_horizon.csv")
cpu = ph[ph.target == "aggregate_cpu_utilization"].set_index("horizon_step")
cpu[["persistence", "ma6", "ma24", "drift", "ridge_direct", "gbm_direct"]].round(4).head(24)
"""),
        md("""
## Operational takeaway

The honest deliverable on 110 points of a nearly unpredictable series is a strong
baseline plus calibrated uncertainty. Complexity buys nothing here, and the
validation harness is what proves it rather than the author's opinion.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells)


def notebook_03():
    cells = [
        md("""
# 03 - The submission, the one-shot score, and what it means

This notebook is the only place you should read holdout-derived numbers, and it
reads them from the frozen artifact produced by `scripts/final_score.py`, which
ran exactly once.
"""),
        code(SETUP),
        md("""
## The submission

Three columns, 24 rows, timestamps inferred from the data.
"""),
        code("""
preds = pd.read_csv(ROOT / "predictions.csv")
print(preds.head(3).to_string(index=False))
print("...")
print(preds.tail(3).to_string(index=False))
print("rows:", len(preds), "unique timestamps:", preds.timestamp.nunique())
"""),
        md("""
Every value is the same because the selected model is persistence on a series
whose training mean is flat. That is not a bug; it is what the validation chose.

## One-shot scores

Reported as-is. The submitted model lost to a moving average on the holdout.
"""),
        code("""
scores = pd.read_csv(ART / "final_scores.csv")
scores.sort_values(["target", "rmse"]).round(4).to_string(index=False)
"""),
        md("""
Read that table twice. On the training folds persistence beat `ma24`
(0.2123 vs 0.2927). On the 24 holdout steps `ma24` beat persistence
(0.1537 vs 0.1848), a skill score of about -0.20 for the submission.

We did not retune. `docs/protocol.md` says the holdout is scored once and the
model is not changed afterwards, and a result you can only trust when it is
favourable is not a result.

## Why it happened

The holdout window drifted slightly upward relative to the last training day. A
24-step average sits closer to that level than the last single observation does.
That is luck, not skill - `ma24` had no way to know. With five overlapping
origins the standard error was wider than the gap between candidates, so the fold
ranking never had the resolution to separate them.

## Post-hoc analysis (labelled)

Everything below is post-hoc: it explains the result and changes nothing.
"""),
        code("""
print(json.dumps(json.loads((ART / "posthoc_summary.json").read_text()), indent=2)[:1500])
"""),
        code("""
print("errors by horizon (post-hoc):")
print(pd.read_csv(ART / "posthoc_errors_by_horizon.csv").round(4).to_string(index=False))
print()
print("interval coverage (post-hoc):")
print(pd.read_csv(ART / "posthoc_interval_coverage.csv").round(4).to_string(index=False))
"""),
        md("""
The uncertainty statement held up better than the point forecast: the nominal 90%
band covered 91.7% of holdout steps, and every miss was above the band (none
below), consistent with the small upward bias in the forecast.

## The capacity decision

A forecast is not a decision. Provisioning uses a quantile of the band, chosen by
the cost of being short versus long.
"""),
        code("""
print(json.loads((ART / "policy_choice.json").read_text()))
print()
print(pd.read_csv(ART / "policy_holdout_evaluation.csv").round(4).to_string(index=False))
"""),
        code("""
sens = pd.read_csv(ART / "policy_holdout_sensitivity.csv")
sens.pivot_table(index="cost_ratio", columns="policy", values="net_cost_per_step").round(3)
"""),
        md("""
The optimal provisioning level moves from the 50th percentile at a 1:1 cost ratio
to the 85th at 32:1. There is no universal headroom rule; there is only the rule
your cost structure implies. Saying that out loud is more useful than quoting a
single "safe" percentage.

## Operational takeaway

1. The submitted forecast is beatable, and we say so.
2. The uncertainty band was better calibrated than the point forecast and is what
   the capacity decision actually uses.
3. The mean is 9.4 pp while the cross-VM P95 is 37.5 pp, so any plan built on the
   aggregate alone under-provisions the tail.
"""),
    ]
    return nbf.v4.new_notebook(cells=list(cells))


BUILDERS = {
    "01_data_and_eda.ipynb": notebook_01,
    "02_validation.ipynb": notebook_02,
    "03_submission_and_results.ipynb": notebook_03,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the notebooks in place")
    args = parser.parse_args(argv)

    NB_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, builder in BUILDERS.items():
        nb = builder()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        path = NB_DIR / name
        nbf.write(nb, str(path))
        paths.append(path)
        print("wrote", path.relative_to(ROOT))

    if args.execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
        for path in paths:
            nb = nbf.read(str(path), as_version=4)
            ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
            nbf.write(nb, str(path))
            print("executed", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
