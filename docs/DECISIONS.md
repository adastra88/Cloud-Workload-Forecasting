# Decisions log

Chronological. Every non-trivial decision, with the alternatives considered and
the reasoning. Plain English so a newcomer can follow it.

---

## D001 — Timestamps stay in raw seconds (2026-10-26, Phase 0)

**Decision.** `timestamp` in `predictions.csv` is written in the same raw integer
format as the source (seconds relative to the trace start), not converted to a
datetime string.

**Alternatives.** Convert to ISO-8601 using an arbitrary epoch. Rejected: the
epoch is not given in the release, so any conversion is an invention, and
SUBMISSION.md requires the timestamps to "match the holdout timestamps exactly".

**Why it matters.** A reviewer diffing our timestamps against the release's last
24 values gets an exact match.

## D002 — Validity definition for a "valid CPU observation" (2026-10-26, Phase 0)

**Decision.** Valid = `avg_cpu` finite and in `[0, 100]`. Rows are de-duplicated
on `(vm_id, timestamp)` by averaging before counting.

**Alternatives.** Use `min_cpu`/`max_cpu` consistency too (rejected: adds rules
that are not needed by the contract and would make the smoke dataset fussier);
drop duplicates keeping the first (rejected: arbitrary).

**Why it matters.** The definition doubles as the detection mechanism for the
holdout-corruption test — randomised or removed holdout readings must look
"invalid" or absent, and the predictions must not move.

## D003 — No resampling; cadence inferred (2026-10-26, Phase 0)

**Decision.** The aggregate series is indexed by observed timestamps only. The
cadence is the modal gap, inferred at run time.

**Alternatives.** `resample('5min')` with forward-fill (rejected: invents target
values, which DATASET.md explicitly discourages) or with NaN (rejected: leaves
holes that the model then has to handle anyway, and hides the problem).

**Why it matters.** The public release has a perfect 300 s grid, so the choice is
a no-op there; on a gappy smoke dataset it means we never fabricate a target.

## D004 — Split window: 110 train / 24 holdout, with graceful shrink (2026-10-26, Phase 0)

**Decision.** `train_len=110`, `holdout_len=24` are CLI defaults. The split
function shrinks the window rather than raising when the series is short.

**Why it matters.** Reviewers swap in a tiny dataset; a hard-coded 110 would
crash. The defaults document the assignment contract; the fallback keeps the
pipeline honest and useful.

## D005 — Metadata is optional and never a model feature (2026-10-26, Phase 0)

**Decision.** `predict` works from `vm_cpu_readings.csv.gz` alone. Metadata is
loaded if present for descriptive analysis only.

**Alternatives.** Use `vm_category` / core bucket as a one-hot feature on the
aggregate series. Rejected: at the aggregate level these are constant across
timestamps (the population is fixed), so they cannot explain a time series;
and the only time-varying metadata columns are exactly the banned ones.

**Why it matters.** It removes a whole class of leakage risk and keeps the
smoke-dataset path identical to the real path.

## D006 — Rolling-origin design: 5 origins, step 6, min train 60 (2026-10-26, Phase 0)

**Decision.** Origins `{60, 66, 72, 78, 84}`, horizon 24, expanding window.

**Alternatives.** Step 24 (only 2 origins — too few to estimate spread); step 1
(19 heavily overlapping origins — pseudo-replication, over-confident spread).

**Why it matters.** Step 6 balances evidence and independence. Every decision
later references the origin-to-origin spread, never a single fold.

## D007 — Selection rule includes a 1-SE simplicity rule (2026-10-26, Phase 0)

**Decision.** Lowest rolling-origin mean RMSE wins, but any candidate within one
standard error of the best is considered tied and the simpler one is chosen.

**Why it matters.** With 5 overlapping folds, differences of a few percent are
noise. Complexity has to earn its place.

## D008 — Active VM count is expected to be (near-)constant (2026-10-26, Phase 0)

**Decision.** We will check this in Phase 1. If it is constant, the forecast is
the constant (persistence is then exact) and we spend no effort on modelling it;
the pipeline still supports varying counts because the smoke datasets exercise
them.

**Reasoning.** The DATA_CARD states every selected VM has exactly 134 readings at
a complete grid, which implies every VM is present at every timestamp. This is a
property of the *curated release*, not of the underlying fleet — an important
caveat to state in the report rather than paper over.

## D009 — Pin Python 3.12 + pandas 2.2.x (2026-10-26, Phase 0/1)

**Decision.** Build the environment on CPython 3.12 with `pandas>=2.2,<2.3`.

**Alternatives and why rejected.** The first venv was Python 3.14 with pandas
3.1.0 (a dev build). In that wheel `DataFrame.merge()` returns a *tuple*
instead of a DataFrame, which silently broke the EDA joins with a confusing
`KeyError: 'vm_id'`. Python 3.14 has no pandas 2.2 wheel, and building it needs
an MSVC toolchain we do not need for this task. Downgrading the interpreter is
cheaper and closer to what a reviewer will run.

**Consequence.** The README pins `python>=3.10,<3.14`-compatible versions and the
`Makefile setup` target creates the venv with a pinned interpreter.

## D010 — `active_vm_count` confirmed constant; forecast by persistence (2026-10-26, Phase 1)

**Decision.** Do not model `active_vm_count`. Use the last observed value as the
forecast for all 24 steps, with a documented constant fallback for degenerate
inputs.

**Evidence.** Over the 110 training timestamps the count is exactly 10,000 at
every one, and 0% of VMs are deleted inside the window
(`reports/artifacts/target_summary_train.csv`, `churn_by_category.csv`).

**Honesty note.** This is a property of the curated release, not of Azure. The
interesting question — for how long does a VM stay alive, how do arrivals and
departures move the count — cannot be answered with a complete rectangular
slice. We say so instead of inventing a model. The code path still handles
varying counts (smoke datasets + tests), because a reviewer may swap one in.

## D011 — No seasonal component in any candidate (2026-10-26, Phase 1)

**Decision.** `seasonal_naive` is evaluated only as a diagnostic, never selected.

**Evidence.** The training window is 9.2 h (110 points on a 5-minute grid).
There is no daily or weekly cycle to estimate — lag-24 autocorrelation is
−0.02, inside the 95% band — and any imposed period would be fitted to noise.
The ACF falls inside the band by lag 11 (~55 min), so even an hourly period is
weakly supported. We log the diagnostic and move on.
