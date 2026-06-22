# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install train recommend evaluate test lint format clean

help:
	@echo "Targets:"
	@echo "  install    Install the package + dev extras (pip)"
	@echo "  train      Fit ALS factors from the configured ratings"
	@echo "  recommend  Write top-N recommendations for users"
	@echo "  evaluate   Score ALS vs the popularity baseline (ranking metrics)"
	@echo "  test       Run pytest (pure-numpy known-answer tests always run)"
	@echo "  lint       ruff check + mypy"
	@echo "  format     ruff format"
	@echo "  clean      Remove caches and build artefacts"

install:
	pip install -e ".[dev,spark]"

train:
	recsys train --config config/movielens.yaml

recommend:
	recsys recommend --config config/movielens.yaml

evaluate:
	recsys evaluate --config config/movielens.yaml

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
