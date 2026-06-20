# eo-monitor — uniform verbs: run, test, lint
# Uses pixi if available; falls back to the active Python environment otherwise.

CONFIG ?= config/corn_belt.yaml
PIXI := $(shell command -v pixi 2>/dev/null)

.PHONY: run test lint format install help

help:
	@echo "Targets:"
	@echo "  install  install the package + dev deps (pixi if present, else pip)"
	@echo "  run      run the pipeline with CONFIG=$(CONFIG)"
	@echo "  test     run the pytest suite"
	@echo "  lint     run ruff check + mypy"
	@echo "  format   apply ruff formatter"

install:
ifdef PIXI
	pixi install
else
	pip install -e ".[dev]"
endif

run:
ifdef PIXI
	pixi run eo-monitor run --config $(CONFIG)
else
	eo-monitor run --config $(CONFIG)
endif

test:
ifdef PIXI
	pixi run pytest
else
	pytest
endif

lint:
ifdef PIXI
	pixi run ruff check src tests
	pixi run mypy
else
	ruff check src tests
	mypy
endif

format:
ifdef PIXI
	pixi run ruff format src tests
else
	ruff format src tests
endif
