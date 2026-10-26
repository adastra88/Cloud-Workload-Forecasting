# Progress log

Updated after every phase so a fresh session can resume.

## Phase 0 — Setup and protocol freeze ✅ (2026-10-26)

**Done**
- Read upstream `README.md`, `ASSIGNMENT.md`, `DATASET.md`, `SUBMISSION.md`,
  `DATA_RELEASE.md`, `SOURCES.md`, starter `src/cwfc/*`, bundled `DATA_CARD.md`.
- Downloaded the fixed `data-v1` release asset (`data-v1.tar.gz`,
  116,513,018 bytes). Asset SHA-256 matches the release metadata
  (`d04d402a…9caeb`). Extracted to `challenge_data/`; all three files in
  `SHA256SUMS` verified OK.
- Structural sanity check: 1,340,000 rows, 10,000 VMs × 134 readings, 134
  distinct timestamps, every gap exactly 300 s, no nulls, no duplicate
  `(vm_id, timestamp)`, no `avg_cpu` outside [0,100], `min_cpu <= max_cpu`
  everywhere.
- Python 3.12 venv at `.venv/` (see D009), deps installed.
- Wrote and committed `docs/protocol.md`, `docs/feature_audit.md`,
  `docs/DECISIONS.md` **before running any model**.

**Next**: Phase 1 training-window EDA.

**Open questions**: none blocking.

## Phase 1 — Data understanding (training window only) ✅

**Done**
- `scripts/run_eda.py` runs end to end and writes 20 artifacts to
  `reports/artifacts/` plus figures to `reports/figures/`. Structural checks
  (schema, dtypes, grid, duplicates, counts) run over the full file and never
  print a target value; every pattern-finding section is restricted to the 110
  training timestamps.
- Findings: `active_vm_count` is exactly constant (10,000); no in-window churn;
  aggregate CPU mean 9.41 pp with std 0.19 pp; no trend (ADF p=0.41, KPSS
  p=0.10); ACF decays inside the 95% band by lag 11 (≈55 min); 83.6% of
  variance is between-VM level, 0.018% common-time; movement is 100% within-VM
  (composition effect exactly zero because no VM enters or leaves); aggregate
  mean 9.4 vs cross-VM P95 37.5 vs max 99.3 (mean hides tails); top 1,000 VMs
  carry 43.5% of load; peak threshold (train top decile) 9.67 pp.
- `docs/journal/phase1.md` written.

**Next**: Phase 2 — targets extras and split tests; then Phase 3 validation
harness and baselines.

**Open questions**: none blocking.

## Phase 2 — Targets and split ✅

**Done**
- `src/cwf/targets.py` builds both required targets plus three clearly labelled
  extras (`share_hot_vms`, `cross_vm_p95_avg_cpu`, `aggregate_max_cpu`).
  Validity: `avg_cpu` must be finite and within [0,100]; a VM is counted at most
  once per timestamp; missing readings are excluded from the mean, never
  zero-filled.
- `src/cwf/splits.py` owns the only split: 110 train / 24 holdout inferred from
  the data, plus `train_readings`, the guard used by every in-sample analysis.
- Split tests pass (sizes, chronological, no overlap, single ownership).

## Phase 3 — Validation harness and baselines ✅

**Done**
- `src/cwf/metrics.py`, `src/cwf/baselines.py` (persistence, moving average
  6/12/24, drift, seasonal naive), `src/cwf/evaluate.py` with an expanding-window
  rolling-origin evaluator: origins `[60, 66, 72, 78, 84]`, horizon 24, every
  fold satisfies `origin + horizon <= 110`, plus a dev-holdout fold at origin 85.
- Baseline RMSE on `aggregate_cpu_utilization`: persistence **0.2123**
  (SE 0.0425), drift 0.2396, ma6 0.2683, ma12 0.2798, ma24 0.2927.
  `active_vm_count` is exactly constant, so every candidate scores RMSE 0.0000
  and MASE is undefined (zero naive scale) — reported honestly as such.
- Confirms the expectation: the naive baseline is very hard to beat on 110
  points of a stationary, noisy series.

## Phase 4 — Models ✅

**Done**
- `src/cwf/features.py` builds origin-safe lag/rolling/shape features from the
  training context only (causality is structural: the builder never receives
  future values).
- `src/cwf/models.py`: damped-trend ETS, Holt ETS, direct multi-horizon ridge on
  lag features (alpha sweep 0.1→1000), direct gradient boosting, constant model.
- Rolling-origin RMSE on `aggregate_cpu_utilization`: persistence **0.2123**,
  gbm_direct 0.2209, ets_damped 0.2842, ridge_direct 0.2884 (alpha 10; the best
  alpha 0.1 gave 0.2615, still worse), ets_holt 0.2953. With ~50-60 training
  rows and 5 overlapping origins, differences inside the 1-SE band
  (persistence, drift, gbm) are not separable.

## Phase 5 — Selection, final fit, predict command ✅

**Done**
- `docs/protocol.md` fixed the rule in advance (lowest rolling-origin RMSE,
  1-SE simplicity tie-break). `scripts/select_model.py` applied it and froze
  `config/final_model.json` (persistence for both targets).
- `python -m cwf.predict --data challenge_data --output predictions.csv` builds
  the series, splits, discards the holdout **values**, refits on train only,
  forecasts 24 steps, maps to the inferred holdout timestamps and clips.
- `predictions.csv`: 24 rows, exactly
  `timestamp,active_vm_count,aggregate_cpu_utilization`. Passes
  `scripts/validate_submission.py`.
- Holdout-corruption invariance (R9): replaced holdout readings with random
  values and with zeros → byte-identical predictions (sha256 `0bc68527…d5939`).

## Phase 6 — One-shot scoring and post-hoc analysis ✅

**Done**
- `scripts/final_score.py` is the single place holdout truth is loaded; it ran
  exactly once and its output is frozen in `reports/artifacts/final_scores.csv`.
  **No retuning followed.**
- Result reported as-is: `aggregate_cpu_utilization` — submitted persistence
  RMSE 0.1848, MAE 0.1468, MASE 1.2119, skill vs best baseline −0.2024.
  `ma24` scored RMSE 0.1537 / MASE 0.9652, i.e. **the baseline won on the
  holdout** even though persistence won on the training folds. Honest headline:
  five overlapping origins of a noisy series did not rank the candidates
  reliably, and the pre-registered rule was still followed.
- Post-hoc (labelled, does not change the model): holdout mean actual 9.2690 vs
  forecast 9.1646 (bias +0.1044); zero holdout steps above the training peak
  threshold 9.6684; interval coverage at nominal 50/80/90/95 was
  0.833/0.833/0.917/0.917 with all misses above the band; no churn in the
  holdout window.

## Phase 7 — Operational recommendation ✅

**Done**
- `src/cwf/policy.py`: newsvendor-style provisioning on the forecast band, with
  the cost asymmetry (`c_under`/`c_over`) an explicit parameter.
  `scripts/run_policy.py` calibrates the band on rolling-origin residuals and
  chooses the level on folds only: `widen_factor = 3.0`, chosen level 0.60,
  peak trigger "hold extra headroom".
- `scripts/evaluate_policy.py` applies the frozen policy to the holdout once:
  at cost ratio 8:1 the level-0.60 policy gives 12 shortfall steps out of 24,
  mean over-provisioning 0.048 pp, net cost 0.611 per step. The sensitivity
  table spans cost ratios 1→32 and moves the optimum level 0.50→0.85: the
  margin should follow the cost asymmetry, not a fixed rule of thumb.

**Next**: Phase 8 packaging (README, Makefile, notebooks, full test suite,
clean-clone reproduction).
