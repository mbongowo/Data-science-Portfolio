# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install demo ingest score trends test lint format clean

help:
	@echo "Targets:"
	@echo "  install   Install the package + dev extras (pip)"
	@echo "  demo      Run the seeded synthetic end-to-end demo (numpy/pandas only)"
	@echo "  ingest    Load compressed Reddit dumps -> cleaned Parquet"
	@echo "  score     Score documents (lexicon or model) -> Parquet"
	@echo "  trends    Aggregate scores into a daily/weekly sentiment series"
	@echo "  test      Run pytest (pure-numpy/pandas known-answer tests always run)"
	@echo "  lint      ruff check + mypy"
	@echo "  format    ruff format"
	@echo "  clean     Remove caches and build artefacts"

install:
	pip install -e ".[dev,spark,model]"

demo:
	sentiment demo --out outputs

ingest:
	sentiment ingest --config config/reddit.yaml --out data/raw

score:
	sentiment score --config config/reddit.yaml --data data/raw --out outputs

trends:
	sentiment trends --config config/reddit.yaml --data outputs --out outputs

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
