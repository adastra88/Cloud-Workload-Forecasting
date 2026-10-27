# Forecasting protocol (frozen before any model was evaluated)

Status: **FROZEN** at the Phase 0 commit. Everything below was fixed *before* a
single candidate model was run on the data. Later edits are appended to
`docs/DECISIONS.md` and never silently change this file. If any rule had to
change, the change is logged with its date and reason.

---

## 1. Task restatement

Build the two required aggregate series from the `data-v1` release, fit on the
first 110 observations, and forecast the final 24 timestamps (2 hours at 5-minute
cadence):

* `active_vm_count(t)`
* `aggregate_cpu_utilization(t)`

The holdout values are present in the public file. Correctness therefore depends
on discipline, not on a hidden-label service.

## 2. Target definitions

A **valid reading** is one where `avg_cpu` is finite and within `[0, 100]`.
`min_cpu`/`max_cpu` are not used to define validity; they are used only in
descriptive analysis and in the optional `aggregate_max_cpu` extra target.

* `active_vm_count(t)` = number of distinct `vm_id` with a valid reading at `t`.
  Duplicate `(vm_id, timestamp)` rows are de-duplicated first (mean of
  duplicates), so a VM is counted at most once per timestamp.
* `aggregate_cpu_utilization(t)` = arithmetic mean of the valid VM-level
  `avg_cpu` at `t`. Missing or invalid readings are **excluded**, never
  zero-filled. A timestamp with zero valid readings gets `NaN` and the pipeline
  aborts with a clear message rather than inventing a number. (On the public
  release this never occurs; tests assert it.)
* Timestamp units are the raw source unit (seconds relative to the trace). The
  cadence used for reporting is inferred from the data (modal gap), never
  hard-coded.
* No resampling. The series is indexed by observed timestamps only. Gaps, if
  any, are reported and left as gaps; the model still receives a chronologically
  ordered series.

Optional **extra** targets (clearly labelled "extras" wherever used):
`share_hot_vms` (fraction of active VMs with `avg_cpu` >= threshold),
`cross_vm_p95_avg_cpu`, `aggregate_max_cpu`. The high-CPU `threshold` is fixed
from the **training window only** (90th percentile of training-window VM-level
`avg_cpu` readings).

## 3. Split

One function: `cwf.splits.chronological_split`.

* Full aggregate series built first, then split.
* `holdout_len = 24` (CLI default).
* `train_len = 110` (CLI default): the training window is the last 110
  observations before the holdout. If the series is too short, the window
  shrinks; if there is not enough history for the chosen model the predictor
  falls back to a baseline and logs a warning.
* Holdout timestamps are the last `holdout_len` distinct timestamps in the data.
* Predictions are produced for exactly those timestamps, in chronological order.

## 4. Leakage rules (absolute)

| # | Rule |
|---|------|
| R1 | Only `predict`/`final_score` touch holdout rows; `predict` keeps only holdout *timestamps*. |
| R2 | No holdout target values are printed, plotted or summarised before the one-shot scoring. |
| R3 | Pattern-finding EDA uses the training window only; only structural checks run over the full file. |
| R4 | Banned features: `timestamp_vm_deleted`, metadata `max_cpu`, metadata `avg_cpu`, metadata `p95_max_cpu`. `timestamp_vm_created` is allowed only as "age at origin" for VMs created at or before the origin. |
| R5 | All fitting, feature choice, tuning, model selection and interval calibration happen inside the training window via rolling-origin evaluation. |
| R6 | This document precedes any model result. |
| R7 | Holdout truth is loaded in exactly one place: `scripts/final_score.py`, run once after freezing. |
| R8 | Post-hoc holdout analysis is allowed after R7 and is labelled "post-hoc". |
| R9 | Corrupting (randomising / removing / zeroing) holdout readings must not change `predictions.csv`. |
| R10 | Any suspected violation is logged in `docs/DECISIONS.md` and the report. |

**Causality contract.** A feature value computed at forecast origin `o` may read
only data with `timestamp <= o`. Enforced by `cwf.features` (every feature builder
receives a truncated frame) and tested by a property test that perturbs the
future and asserts the past features are unchanged.

## 5. Validation scheme

**Rolling-origin, expanding window, single horizon window per origin.**

* Training window length: `110` observations, ending at the origin.
* Origin step: `6` observations (30 minutes), giving overlapping 24-step
  evaluation windows and therefore more origins than a step of 24.
* Horizon: `24`.
* Constraint: `origin_index + 24 <= 110`. Origins are enumerated
  programmatically as `range(min_train, n - horizon + 1, step)` with
  `min_train = 60`; on 110 observations that yields origins
  `{60, 66, 72, 78, 84}`, i.e. **5 origins**, with expanding training windows of
  60-84 observations. The exact indices are saved to
  `reports/artifacts/validation_folds.csv`.
* Minimum training length: `60` observations (5 hours). Rationale: 60 points
  allow the damped-trend optimiser and the 12-lag ridge features to be
  estimated without the fit being dominated by a handful of observations.
* Every candidate, baselines included, goes through the identical harness.

**"Dev holdout" fold** (mirrors the final protocol once): train on indices
`0..85`, evaluate on the last 24 of the 110 (indices `86..109`). Used only as a
consistency check; the rolling-origin mean is the selection criterion.

**Metrics** per target, per origin and aggregated:

* MAE, RMSE (primary: RMSE, in the target's own units: percentage points for
  utilisation, VMs for the count).
* MASE, scaled by the **in-sample** naive (one-step) MAE of the training window
  of that origin, computed from training data only.
* Skill score `1 - MSE_model / MSE_baseline` against the best baseline.
* Per-horizon error curve (24 values) and per-origin spread.

Reference baselines for the skill score: `persistence`, `moving_average(6|12|24)`,
`drift`. "Best baseline" = the baseline with the lowest rolling-origin mean RMSE
for that target, chosen on validation data only.

## 6. Candidate list (fixed before evaluation)

Per target:

| id | family | key hyper-parameters |
|----|--------|----------------------|
| `persistence` | baseline | - |
| `ma6`, `ma12`, `ma24` | baseline | window in {6, 12, 24} |
| `drift` | baseline | - |
| `seasonal_naive` | baseline | season = 24 **only if** Phase-1 autocorrelation justifies it; decided inside the training window and recorded |
| `ets_damped` | statistical | additive error, additive damped trend, no seasonality; optimised by MLE |
| `ets_holt` | statistical | additive error, additive trend (no damping) |
| `ridge_direct` | linear ML | direct multi-horizon, one model per horizon step pooled across origins; features = origin-safe lags/rolling stats/growth; heavy L2; single fixed `alpha` grid chosen on the dev-holdout fold only |
| `gbm_direct` | boosting | direct multi-horizon; `HistGradientBoostingRegressor`, depth <= 3, small learning rate, early stopping off with fixed small trees |

Rejected up front: LSTM/Transformer (110 training points; no defensible reason),
recursive ML (error accumulation with no evidence it helps), calendar features
(11 hours of data cannot identify daily or weekly cycles; tested honestly and
reported even if they look "significant" by chance), per-VM global model
(possible stretch goal, only after everything else is done).

## 7. Selection rule (pre-registered)

1. Compute rolling-origin mean RMSE for every candidate on both targets.
2. Pick the candidate with the **lowest mean RMSE** per target.
3. **One-standard-error (1-SE) simplicity rule**: across origins compute the
   standard error of the per-origin RMSE. Any candidate whose mean RMSE is within
   `1 x SE_of_best` of the best is considered statistically indistinguishable;
   among those, pick the **simplest** by this fixed order:
   `persistence < ma6 < ma12 < ma24 < drift < seasonal_naive < ets_holt <
   ets_damped < ridge_direct < gbm_direct`.
4. If the selected model is a baseline, say so plainly and submit the baseline.
5. The selection output is written to `config/final_model.json` (with the
   metrics that justified it) and then frozen.
6. The holdout is touched **once** afterwards, by `scripts/final_score.py`.

Rationale: with about 5 overlapping origins the evidence is noisy; always
preferring the complex model on a tie would be a form of overfitting to the
validation set.

## 8. Prediction intervals

Empirical residual quantiles by horizon step, from the same rolling-origin
residuals used for selection (quantiles 0.05/0.10/0.50/0.90/0.95), with a
`predicted +/- q` construction on the selected model. Bands are written to
`reports/artifacts/forecast_intervals.csv`, never into `predictions.csv`
(schema allows three columns only). Coverage is measured on the rolling-origin
folds; one-shot holdout coverage is reported post-hoc.

## 9. Capacity policy (fixed before seeing holdout)

Provision `q_target = ceil(quantile_level)` at an asymmetric-cost-optimal
quantile derived from `c_under / c_over` (classic newsvendor ratio
`q = c_under / (c_under + c_over)`); the quantile level is chosen on
rolling-origin folds, then evaluated **once** on the holdout.
`c_under = 8` (cost of one VM-share of shortfall), `c_over = 1`; sensitivity
table over `c_under/c_over` in {1, 2, 4, 8, 16, 32}.
A peak-risk trigger warns when `P(utilisation > train peak threshold) > 0.25` or
the band width exceeds 2x the training-window median absolute one-step change.
The mean is not the tail: the policy is evaluated on the aggregate mean, and the
hot-VM extra target is used to explain why a mean-based rule under-protects.

## 10. Reproducibility

Python >= 3.10, versions pinned in `pyproject.toml` lower bounds and recorded in
`reports/artifacts/environment.txt`. Fixed seeds (`SEED = 0`) everywhere a
random number generator is used. Deterministic ordering (`sort_values` with
stable mergesort). One command reproduces the submission:

```bash
python -m cwf.predict --data challenge_data --output predictions.csv
```

## 11. What is reported even if it is unflattering

Every metric table shows baselines and candidates side by side on identical
folds, including the case where a simple baseline wins. Gains smaller than the
origin-to-origin spread are described as "not distinguishable from noise".