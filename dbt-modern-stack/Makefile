# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install seed build dq test lint format clean

help:
	@echo "Targets:"
	@echo "  install   Install the package + dbt + dev extras (pip)"
	@echo "  seed      Load raw IMDb extracts into the DuckDB warehouse"
	@echo "  build     dbt build (run all models + generic/singular tests)"
	@echo "  dq        Run the pure-pandas data-quality core"
	@echo "  test      Run pytest (pure-pandas known-answer tests always run)"
	@echo "  lint      ruff check + mypy"
	@echo "  format    ruff format"
	@echo "  clean     Remove caches and build artefacts"

install:
	pip install -e ".[dbt,dev]"

seed:
	dwh seed --config config/warehouse.yaml

build:
	dbt build --project-dir transform --profiles-dir transform

dq:
	dwh dq --config config/warehouse.yaml

test:
	pytest

lint:
	ruff check .
	mypy src

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	rm -rf transform/target transform/dbt_packages transform/logs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
