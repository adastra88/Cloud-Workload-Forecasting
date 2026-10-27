# Decisions log

Chronological. Every non-trivial decision, with the alternatives considered and
the reasoning. Plain English so the reasoning is auditable.

---

## D001 - Timestamps stay in raw seconds (2026-10-26, Phase 0)

**Decision.** `timestamp` in `predictions.csv` is written in the same raw integer
format as the source (seconds relative to the trace start), not converted to a
datetime string.

**Alternatives.** Convert to ISO-8601 using an arbitrary epoch. Rejected: the
epoch is not given in the release, so any conversion is an invention, and
SUBMISSION.md requires the timestamps to "match the holdout timestamps exactly".

**Why it matters.** A reviewer diffing the timestamps against the release's last
24 values gets an exact match.

## D002 - Validity definition for a "valid CPU observation" (2026-10-26, Phase 0)

**Decision.** Valid = `avg_cpu` finite and in `[0, 100]`. Rows are de-duplicated
on `(vm_id, timestamp)` by averaging before counting.

**Alternatives.** Use `min_cpu`/`max_cpu` consistency too (rejected: adds rules
that are not needed by the contract and would make the smoke dataset fussier);
drop duplicates keeping the first (rejected: arbitrary).

**Why it matters.** The definition doubles as the detection mechanism for the
holdout-corruption test: randomised or removed holdout readings must look
"invalid" or absent, and the predictions must not move.

## D003 - No resampling; cadence inferred (2026-10-26, Phase 0)

**Decision.** The aggregate series is indexed by observed timestamps only. The
cadence is the modal gap, inferred at run time.

**Alternatives.** `resample('5min')` with forward-fill (rejected: invents target
values, which DATASET.md explicitly discourages) or with NaN (rejected: leaves
holes that the model then has to handle anyway, and hides the problem).

**Why it matters.** The public release has a perfect 300 s grid, so the choice is
a no-op there; on a gappy smoke dataset it means the code never fabricates a
target.

## D004 - Split window: 110 train / 24 holdout, with graceful shrink (2026-10-26, Phase 0)

**Decision.** `train_len=110`, `holdout_len=24` are CLI defaults. The split
function shrinks the window rather than raising when the series is short.

**Why it matters.** Reviewers swap in a tiny dataset; a hard-coded 110 would
crash. The defaults document the assignment contract; the fallback keeps the
pipeline honest and useful.

## D005 - Metadata is optional and never a model feature (2026-10-26, Phase 0)

**Decision.** `predict` works from `vm_cpu_readings.csv.gz` alone. Metadata is
loaded if present for descriptive analysis only.

**Alternatives.** Use `vm_category` / core bucket as a one-hot feature on the
aggregate series. Rejected: at the aggregate level these are constant across
timestamps (the population is fixed), so they cannot explain a time series;
and the only time-varying metadata columns are exactly the banned ones.

**Why it matters.** It removes a whole class of leakage risk and keeps the
smoke-dataset path identical to the real path.

## D006 - Rolling-origin design: 5 origins, step 6, min train 60 (2026-10-26, Phase 0)

**Decision.** Origins `{60, 66, 72, 78, 84}`, horizon 24, expanding window.

**Alternatives.** Step 24 (only 2 origins, too few to estimate spread); step 1
(19 heavily overlapping origins, pseudo-replication, over-confident spread).

**Why it matters.** Step 6 balances evidence and independence. Every decision
later references the origin-to-origin spread, never a single fold.

## D007 - Selection rule includes a 1-SE simplicity rule (2026-10-26, Phase 0)

**Decision.** Lowest rolling-origin mean RMSE wins, but any candidate within one
standard error of the best is considered tied and the simpler one is chosen.

**Why it matters.** With 5 overlapping folds, differences of a few percent are
noise. Complexity has to earn its place.

## D008 - Active VM count is expected to be (near-)constant (2026-10-26, Phase 0)

**Decision.** Check this in Phase 1. If it is constant, the forecast is the
constant (persistence is then exact) and no effort goes into modelling it; the
pipeline still supports varying counts because the smoke datasets exercise them.

**Reasoning.** The DATA_CARD states every selected VM has exactly 134 readings at
a complete grid, which implies every VM is present at every timestamp. This is a
property of the *curated release*, not of the underlying fleet, and an important
caveat to state in the report rather than paper over.

## D009 - Pin Python 3.12 + pandas 2.2.x (2026-10-26, Phase 0/1)

**Decision.** Build the environment on CPython 3.12 with `pandas>=2.2,<2.3`.

**Alternatives and why rejected.** The first venv was Python 3.14 with pandas
3.1.0 (a dev build). In that wheel `DataFrame.merge()` returns a *tuple*
instead of a DataFrame, which silently broke the EDA joins with a confusing
`KeyError: 'vm_id'`. Python 3.14 has no pandas 2.2 wheel, and building it needs
an MSVC toolchain this task does not need. Downgrading the interpreter is
cheaper and closer to what a reviewer will run.

**Consequence.** The README pins `python>=3.10,<3.14`-compatible versions and the
`Makefile setup` target creates the venv with a pinned interpreter.

## D010 - `active_vm_count` confirmed constant; forecast by persistence (2026-10-26, Phase 1)

**Decision.** Do not model `active_vm_count`. Use the last observed value as the
forecast for all 24 steps, with a documented constant fallback for degenerate
inputs.

**Evidence.** Over the 110 training timestamps the count is exactly 10,000 at
every one, and 0% of VMs are deleted inside the window
(`reports/artifacts/target_summary_train.csv`, `churn_by_category.csv`).

**Honesty note.** This is a property of the curated release, not of Azure. The
interesting question (for how long does a VM stay alive, how do arrivals and
departures move the count) cannot be answered with a complete rectangular
slice. Say so instead of inventing a model. The code path still handles
varying counts (smoke datasets and tests), because a reviewer may swap one in.

## D011 - No seasonal component in any candidate (2026-10-26, Phase 1)

**Decision.** `seasonal_naive` is evaluated only as a diagnostic, never selected.

**Evidence.** The training window is 9.2 h (110 points on a 5-minute grid).
There is no daily or weekly cycle to estimate: lag-24 autocorrelation is
-0.02, inside the 95% band, and any imposed period would be fitted to noise.
The ACF falls inside the band by lag 11 (~55 min), so even an hourly period is
weakly supported. Log the diagnostic and move on.

## D012 - Rolling-origin folds: 5 origins, expanding window (2026-10-26, Phase 3)

**Decision.** Evaluate every candidate on origins `[60, 66, 72, 78, 84]` with an
expanding training window, horizon 24, plus a single dev-holdout fold at origin
85 that mirrors the final 1-step-of-24 protocol.

**Alternatives rejected.** (a) A single train/test split: one fold cannot give a
standard error, and it would mean tuning on the same data being reported.
(b) A sliding (windowed) scheme: 110 points are too few to truncate further.
(c) Many more origins (step 1): folds would overlap almost completely,
inflating confidence without adding information.

**Why it matters.** These 5 folds are the ONLY basis for model selection.
Everything downstream (predict, intervals, policy) uses them, so the holdout is
never touched before R7.

## D013 - Selected persistence for both targets (2026-10-26, Phase 5)

**Decision.** `config/final_model.json` freezes `persistence` for
`active_vm_count` and `aggregate_cpu_utilization`.

**Evidence.** Training-fold RMSE on `aggregate_cpu_utilization`: persistence
0.2123 (SE 0.0425), gbm_direct 0.2209, drift 0.2396, ma24 0.2927,
ets_damped 0.2842, ridge 0.2884-0.2953. Persistence, drift and gbm are inside
one standard error; the pre-registered rule breaks the tie toward simplicity.

**Alternatives rejected.** Ridge on lag features (alpha tuned 0.1-1000, best
0.2615, worse than persistence even with the best alpha, and the sweep is
reported so a reviewer can check). GBM (arguably tied, but harder to explain and
not better than the tie-break threshold).

## D014 - Widen the empirical band by 3x (2026-10-26, Phase 7)

**Decision.** `widen_factor = 3.0` on the rolling-origin residual quantiles,
method `empirical`.

**Evidence.** Raw empirical quantiles from 5 origins x 24 steps achieved only
0.558-0.583 leave-one-origin-out coverage at every nominal level, far below
nominal. Widening by 3.0 lifts measured coverage to 0.933. The candidate factors
1.0/1.25/1.5/2.0/2.5/3.0/4.0 are all in
`reports/artifacts/interval_calibration.csv` so the choice is auditable.

**Honest caveat.** Five origins is a tiny calibration sample. The 3x factor is
fitted on the same folds used for selection, so coverage optimism is possible;
the holdout measured 0.917 at nominal 0.90, which is consistent but not proof.

## D015 - The baseline beat the submission on the holdout (2026-10-26, Phase 6)

**Decision.** Report the holdout result as-is. Do not retune, do not re-select,
do not re-freeze.

**What happened.** On the training folds persistence was best (0.2123 vs ma24
0.2927). On the one-shot holdout persistence scored RMSE 0.1848 / MASE 1.2119
while `ma24` scored 0.1537 / 0.9652, a skill score of -0.2024 for the
submission.

**Why not fix it.** R7 forbids changing the model after scoring. The
underlying cause is real and worth stating: five overlapping folds give about 13
correlated evaluations, the SE (0.0425) is larger than the gap between most
candidates, and a 24-step smoothing average happened to sit closer to the
holdout's mild upward drift (holdout mean 9.269 vs train-last-24 mean 9.250 vs
training mean 9.414). This is the honest lesson of the exercise, and it is what
the report leads with.

## D016 - Interval coverage measured post-hoc, not used to re-fit (2026-10-26, Phase 6)

**Decision.** Measure holdout interval coverage once, label it post-hoc, and
leave the widening factor untouched.

**Evidence.** `reports/artifacts/posthoc_interval_coverage.csv`: nominal
0.50/0.80/0.90/0.95 -> measured 0.833/0.833/0.917/0.917. All misses are above
the band (0 below), i.e. the band is slightly high-biased, consistent with the
+0.104 point bias.