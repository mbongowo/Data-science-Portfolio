# access-to-care developer tasks.
# Uses pixi when available (preferred), otherwise plain python/pip.

.PHONY: help install download run test lint format clean

help:
	@echo "Targets: install download run test lint format clean"

install:
	pixi install

download:
	pixi run download

# Build network + access surface (expects data downloaded into data/raw).
run:
	pixi run access data/raw/cameroon-latest.osm.pbf data/raw/healthsites_cameroon.geojson

test:
	pixi run test

lint:
	pixi run lint

format:
	pixi run format

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ outputs/*.gpkg outputs/*.png
