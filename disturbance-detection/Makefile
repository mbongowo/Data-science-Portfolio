.PHONY: install run test lint fmt clean

install:
	pixi install

run:
	pixi run disturb --config config/aoi.yaml

test:
	pixi run pytest

lint:
	pixi run ruff check src tests
	pixi run mypy src

fmt:
	pixi run ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ outputs/*.tif outputs/*.png
