PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: setup test eval report lint

setup:            ## create the virtualenv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e ".[dev]"

test:             ## unit tests
	$(PY) -m pytest

eval:             ## run all golden sets, write reports/, then enforce the quality gate
	$(PY) -m aiq.evals.runner run
	$(PY) -m aiq.evals.runner gate

report:           ## re-render reports/eval_report.md from reports/eval_results.json
	$(PY) -m aiq.report

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
