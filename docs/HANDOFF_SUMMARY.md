# Handoff summary

Plain English. What was asked, what was built, what came out, and what you should
check yourself before you submit.

---

## What was asked

Forecast two things 24 steps ahead (two hours on a five-minute grid) from a
10,000-VM slice of the Microsoft Azure Public Dataset V2:

1. `active_vm_count` — how many VMs are reporting.
2. `aggregate_cpu_utilization` — the mean CPU utilisation across them, in percent.

Deliver a reproducible repo with code, tests, notebooks, a README, a
`predictions.csv`, a report, and — because you'll be interviewed about it — an
interview-prep guide. The grading rubric weights data understanding 15, EDA 20,
methodology 20, model quality 15, error analysis 15, engineering 10,
communication 5. Note what that implies: **model quality is only 15%**. Honest
validation and clear reasoning carry most of the marks.

---

## What was built

- A Python package `cwf` with one job per module: load, build targets, split,
  features, baselines, models, evaluate, metrics, intervals, policy, predict.
- Ten scripts that produce every number in the report.
- 61 tests, all passing.
- Three notebooks that import from `src` and execute top to bottom.
- `predictions.csv`, 24 rows, exactly three columns.
- Documentation: report, this summary, interview guide, decision log,
  per-phase journals, feature audit, pre-registered protocol.

---

## The data in ten bullets

1. 1,340,000 rows: 10,000 VMs × 134 timestamps.
2. Timestamps are a perfect 300-second grid with no gaps and no duplicates.
3. Zero nulls, zero out-of-range `avg_cpu`, zero duplicate keys.
4. `min_cpu <= max_cpu` everywhere.
5. One metadata row per VM.
6. Training window: first 110 timestamps (9.17 hours). Holdout: last 24 (2 hours).
7. `active_vm_count` is **exactly 10,000** at every timestamp — it never moves.
8. `aggregate_cpu_utilization` has mean 9.41%, std 0.19%, range 9.11–9.99%.
9. No trend (ADF p=0.41), no seasonality (lag-24 ACF −0.02), stationarity on the
   boundary (KPSS p=0.10).
10. 83.6% of variance is *which VM you look at*; only 0.018% is a common
    time-varying signal. The aggregate mean is 9.41% while the cross-VM P95 is
    37.55% and the max is 99.30%.

---

## The approach, in three sentences

Pre-register the leakage protocol and the selection rule **before** looking at any
model result; evaluate nine candidates (five baselines, ETS, ridge, gradient
boosting) on five rolling-origin folds entirely inside the training window; apply
the frozen rule, freeze `config/final_model.json`, generate `predictions.csv`, then
score the holdout exactly once and report whatever comes out.

The rule chose **persistence** for both targets.

---

## Headline results, versus baseline

`active_vm_count`: RMSE 0.0000 for the submission and for every baseline. MASE is
undefined because the series doesn't move — dividing by a zero naive scale. Say
that plainly; it's a real property of the data, not a gap in the analysis.

`aggregate_cpu_utilization`, one-shot holdout:

| candidate | RMSE | MAE | MASE | skill vs best baseline |
| --- | --- | --- | --- | --- |
| `ma24` (best baseline) | 0.1537 | 0.1169 | 0.9652 | 0.0000 |
| **submitted (persistence)** | **0.1848** | **0.1468** | **1.2119** | **−0.2024** |

**The baseline won.** Your submission is ~20% worse than a 24-step moving average
on the holdout. This is the single most important thing to be able to explain, and
the explanation is credible: on the training folds persistence was best
(0.2123 vs 0.2927 for `ma24`), but with five overlapping origins the standard
error was 0.0425 — wider than the gap between nearly every pair of candidates. The
fold ranking had no resolution. The holdout also drifted up slightly (mean 9.269
vs 9.250 in the last training day), which favoured a 24-step average by luck.

Changing the model after seeing the holdout would have made the number
meaningless. The protocol says so, and it was followed.

What *did* hold up: the calibrated 90% band covered 91.7% of holdout steps with
zero misses below. The uncertainty statement was better than the point forecast.

---

## The capacity policy

Provision at a quantile of the forecast band, with the cost asymmetry as an
explicit parameter. At 8:1 under-provisioning cost, the newsvendor optimum is the
88.9th percentile; the chosen operating level is 0.60 with a 3× widening on
under-covering empirical bands.

On the holdout at level 0.60: 12 shortfall steps out of 24, mean
over-provisioning 0.048 pp, net cost 0.611 per step. The sensitivity table shows
the optimum moving from the 50th percentile at 1:1 to the 85th at 32:1 — there is
no universal headroom rule, only the one your cost structure implies.

Crucially: **the aggregate mean is not tail risk.** The mean is 9.41% while the
cross-VM P95 is 37.55% and the max is 99.30%. Two machines sit above 90% for about
0.9% of readings. Any plan built on the aggregate alone under-provisions the
machines that matter.

---

## Leakage controls and how each is verified

| control | how it's verified |
| --- | --- |
| Protocol frozen before modelling | `docs/protocol.md` is an earlier commit than any results file; check `git log` |
| Banned metadata columns | every column classified with a reason in `docs/feature_audit.md` |
| One split function | `cwf.splits.chronological_split`; `tests/test_split_contract.py` |
| One holdout reader | `scripts/final_score.py::load_holdout_truth`; `tests/test_leakage.py` parses the AST of every module and fails if anything else references it |
| Forecast blind to holdout values | `tests/test_leakage.py` randomises and zeroes the holdout readings and asserts `predictions.csv` is byte-identical |
| Features don't see the future | `tests/test_causality.py` perturbs everything after an origin and asserts the feature matrix is unchanged |
| One-shot scoring | `scripts/final_score.py` ran once; no retuning followed |
| Post-hoc work labelled | every holdout analysis is in `posthoc_*.csv` / `posthoc_analysis.py` and says so |

---

## What went wrong or surprised me

1. **The baseline won.** Expected as a possibility, still the headline. Handled by
   reporting it, not by hiding it.
2. **The empirical intervals under-covered badly.** Nominal 90% measured 0.558
   across leave-one-origin-out. Fixed by widening 3× (measured 0.933), documented
   with the full calibration table so the choice is auditable.
3. **MASE is undefined for the count.** A zero naive scale. Reported as undefined;
   an unqualified zero would have looked like a perfect model.
4. **The environment fought back.** A pre-release pandas wheel silently returned
   tuples from `DataFrame.merge()`. Fixed by pinning Python 3.12 and pandas 2.2.x.
   Recorded as D009.
5. **Two of my own test assumptions were wrong.** I guessed the signature of
   `build_matrix` and `skill_score`, and initially expected the CLI to reject an
   oversized holdout when it actually shrinks with a warning. The code was right
   and the tests were wrong — which is the tests doing their job, but worth
   knowing so you don't "fix" correct code.

---

## Known weaknesses a skeptical reviewer will flag

- **The model lost to the baseline.** Have the explanation ready (§ above) and do
  not pretend it didn't happen.
- **One holdout window.** A single 24-step evaluation cannot settle a ranking.
  Say so before you're asked.
- **Five origins.** The standard errors are large; treat all fold rankings as
  soft.
- **The count is constant.** It's a property of a rectangular release slice, not
  of Azure. Don't claim you modelled churn — you didn't, because there was none
  to model.
- **The 3× widening factor** is calibrated on the same five origins used for
  selection. Optimistic. Stated in D014.
- **The peak trigger never fired** on the holdout because no step crossed the
  training peak threshold. Correct behaviour, untested behaviour.
- **Length of the window.** 9.2 hours cannot support any claim about daily or
  weekly cycles. Don't make one.

---

## Checklist before you submit

Read:

- [ ] `reports/final_report.md` — all of it. You must be able to say every number.
- [ ] `docs/protocol.md` — it is your defence against "did you tune on the
      holdout?"
- [ ] `docs/feature_audit.md` — know *why* `p95_max_cpu` is banned.
- [ ] `docs/interview/` — start with `00_how_to_use.md`.

Run:

- [ ] `python -m pytest tests` → 61 passed.
- [ ] `python -m cwf.predict --data challenge_data --output predictions.csv` →
      byte-identical to the committed file.
- [ ] `python scripts/validate_submission.py predictions.csv` → OK.
- [ ] `python scripts/build_notebooks.py --execute` → three notebooks, no errors.
- [ ] One smoke dataset by hand (`python -c "from cwf.smoke import *; write_dataset(make_basic(), 'tmp_smoke')"` then predict on it).

Verify yourself:

- [ ] Recompute one metric by hand from `predictions.csv` and the holdout truth.
      Do not skip this — it is the fastest way to catch a misunderstanding.
- [ ] Explain out loud, without notes, why the baseline won.
- [ ] Explain out loud why `timestamp_vm_deleted` cannot be a feature.

---

## Claim → artifact map

| claim | artifact |
| --- | --- |
| 1,340,000 rows, perfect 300 s grid | `reports/artifacts/structure_checks.csv` |
| Count is exactly 10,000 in training | `target_summary_train.csv` |
| Aggregate mean 9.41, std 0.19 | `target_summary_train.csv` |
| No trend, stationarity on the boundary | `stationarity_trend.csv` |
| ACF inside band by lag 11 | `acf_decay.csv` |
| 83.6% between-VM variance | `movement_decomposition.csv` |
| Mean 9.41 vs cross-VM P95 37.55 | `mean_hides_tails.csv` |
| Top 1,000 VMs carry 43.5% | `load_concentration.csv` |
| Peak threshold 9.6684 pp | `peak_threshold.json` |
| Fold origins 60…84 | `validation_folds.csv` |
| Persistence best on folds (0.2123) | `validation_results.csv` |
| Per-origin spread / standard errors | `validation_per_origin.csv` |
| Alpha sweep | `validation_results.csv` + `docs/DECISIONS.md` D013 |
| Frozen selection | `config/final_model.json` |
| Holdout RMSE 0.1848, skill −0.2024 | `final_scores.csv` |
| Holdout bias +0.1044 | `posthoc_summary.json` |
| Interval coverage 0.833/0.833/0.917/0.917 | `posthoc_interval_coverage.csv` |
| Zero holdout peaks | `posthoc_peak_vs_nonpeak.csv` |
| Band widening chosen at 3.0 | `policy_choice.json`, `interval_calibration.csv` |
| Policy holdout cost 0.611/step | `policy_holdout_evaluation.csv` |
| Cost sensitivity 1:1 → 32:1 | `policy_holdout_sensitivity.csv` |
| Extra targets P95 3.99× mean | `extra_targets_train.csv` |
| Zero churn over the full range | `posthoc_churn_context.csv` |
