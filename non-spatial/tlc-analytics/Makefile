# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install demo mart benchmark test lint format clean

help:
	@echo "Targets:"
	@echo "  install    Install the package + dev extras (pip)"
	@echo "  demo       Run the reproducible pandas demo (seeded synthetic data)"
	@echo "  mart       Build the aggregated mart (pandas reference path)"
	@echo "  benchmark  Run the engine bake-off harness"
	@echo "  test       Run pytest (pure-pandas known-answer tests always run)"
	@echo "  lint       ruff check + mypy"
	@echo "  format     ruff format"
	@echo "  clean      Remove caches and build artefacts"

install:
	pip install -e ".[dev,engines]"

demo:
	tlc demo --out outputs

mart:
	tlc mart --config config/tlc.yaml --out outputs

benchmark:
	tlc benchmark --config config/tlc.yaml --out outputs

test:
	pytest

lint:
	ruff check .
	mypy src

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
