# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install test lint format clean pagerank communities triangles

help:
	@echo "Targets:"
	@echo "  install      Install the package + dev extras (pip)"
	@echo "  test         Run pytest (pure-numpy known-answer tests always run)"
	@echo "  lint         ruff check + mypy"
	@echo "  format       ruff format"
	@echo "  clean        Remove caches and build artefacts"
	@echo "  pagerank     Run PageRank from config/graph.yaml -> outputs"
	@echo "  communities  Run connected components + label propagation -> outputs"
	@echo "  triangles    Run triangle counting -> outputs"

install:
	pip install -e ".[dev]"

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

pagerank:
	bdgraph pagerank --config config/graph.yaml --out outputs

communities:
	bdgraph communities --config config/graph.yaml --out outputs

triangles:
	bdgraph triangles --config config/graph.yaml --out outputs
