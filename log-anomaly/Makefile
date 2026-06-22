# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install parse detect evaluate test lint format clean

help:
	@echo "Targets:"
	@echo "  install   Install the package + dev extras (pip)"
	@echo "  parse     Mask + template raw logs into event-count features"
	@echo "  detect    Score sessions with the configured detector (pca / zscore)"
	@echo "  evaluate  Compare flags against HDFS labels (precision/recall/F1)"
	@echo "  test      Run pytest (pure-numpy known-answer tests always run)"
	@echo "  lint      ruff check + mypy"
	@echo "  format    ruff format"
	@echo "  clean     Remove caches and build artefacts"

install:
	pip install -e ".[dev,spark,viz]"

parse:
	loganomaly parse --config config/hdfs.yaml --out outputs

detect:
	loganomaly detect --config config/hdfs.yaml --out outputs

evaluate:
	loganomaly evaluate --config config/hdfs.yaml --out outputs

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
