# Reproducible pipeline.  Targets that write shared artifacts overwrite the
# committed versions, so re-running them should reproduce identical files.
# Windows users can run the underlying commands directly (see README.md).
PY ?= python
DATA ?= challenge_data
OUT ?= predictions.csv

.PHONY: setup data targets eda baselines models select final report test all clean

setup:
	$(PY) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

data:
	$(PY) scripts/download_data.py --dest $(DATA)

eda:
	$(PY) scripts/run_eda.py --data $(DATA)

# Targets, the split and the extra targets are produced by the EDA pass; this
# alias exists so every phase name from the brief resolves to a real command.
targets: eda

# One validation run evaluates every candidate (baselines and models) on the
# identical rolling-origin folds, which is what the protocol requires.
baselines:
	$(PY) scripts/run_validation.py --data $(DATA)

models: baselines

select:
	$(PY) scripts/select_model.py --data $(DATA)
	$(PY) -m cwf.predict --data $(DATA) --output $(OUT)
	$(PY) scripts/validate_submission.py $(OUT)

# One-shot holdout scoring.  Deliberately NOT part of `all`.
final:
	$(PY) scripts/final_score.py --data $(DATA) --predictions $(OUT)

report:
	$(PY) scripts/run_policy.py --data $(DATA)
	$(PY) scripts/evaluate_policy.py --data $(DATA)
	$(PY) scripts/posthoc_analysis.py --data $(DATA)
	$(PY) scripts/run_extras.py --data $(DATA)

test:
	$(PY) -m pytest tests

all: data eda baselines select test

clean:
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
