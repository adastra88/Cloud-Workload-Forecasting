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
