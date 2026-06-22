# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install demo download analyze run test lint format clean

help:
	@echo "Targets:"
	@echo "  install   Install the package + dev extras (pip)"
	@echo "  demo      Run the dependency-free ESDA demo on a seeded grid"
	@echo "  download  Fetch + join USDA NASS yield with TIGER counties"
	@echo "  analyze   Run the ESDA pipeline (alias: run)"
	@echo "  test      Run pytest (pure-numpy known-answer tests always run)"
	@echo "  lint      ruff check + mypy"
	@echo "  format    ruff format"
	@echo "  clean     Remove caches and build artefacts"

install:
	pip install -e ".[dev,gwr]"

demo:
	hotspots demo --seed 0 --out outputs

download:
	python data/download.py --config config/aoi.yaml --out data/raw

analyze run:
	hotspots --config config/aoi.yaml --data data/raw/iowa_corn_yield_2023.gpkg --out outputs

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
