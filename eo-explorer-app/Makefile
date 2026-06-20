# Makefile for eo-explorer-app -- convenience wrappers.
# (Optional: the project also ships pixi tasks; see `pixi run --help`.)

.PHONY: help install install-eo-monitor app test lint format docker clean

help:
	@echo "Targets:"
	@echo "  install            pip install -r requirements.txt + the app (editable)"
	@echo "  install-eo-monitor install the reused flagship package from ../eo-monitor"
	@echo "  app                launch the Streamlit app"
	@echo "  test               run the pytest smoke tests"
	@echo "  lint               ruff check"
	@echo "  format             ruff format"
	@echo "  docker             build the Docker image"
	@echo "  clean              remove caches"

install:
	pip install -r requirements.txt
	pip install -e .

install-eo-monitor:
	pip install -e ../eo-monitor

app:
	streamlit run app/main.py

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

docker:
	docker build -t eo-explorer-app .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
