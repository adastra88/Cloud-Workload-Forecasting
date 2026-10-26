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
- Python 3.14.4 venv at `.venv/`, deps installed.
- Wrote and committed `docs/protocol.md`, `docs/feature_audit.md`,
  `docs/DECISIONS.md` **before running any model**.

**Next**: Phase 1 training-window EDA.

**Open questions**: none blocking.
