# Convenience targets. These assume an activated environment (pixi or venv).
.PHONY: help install bars run report test lint format clean

help:
	@echo "Targets:"
	@echo "  install   Install the package + dev extras (pip)"
	@echo "  bars      Resample raw ticks to OHLCV bars (Parquet)"
	@echo "  run       Run the no-look-ahead backtest -> outputs/equity.parquet"
	@echo "  report    Write the performance report from an equity curve"
	@echo "  test      Run pytest (pure numpy/pandas known-answer tests)"
	@echo "  lint      ruff check + mypy"
	@echo "  format    ruff format"
	@echo "  clean     Remove caches and build artefacts"

install:
	pip install -e ".[dev,scale]"

bars:
	backtest bars --config config/strategy.yaml --ticks data/raw/BTCUSDT-trades.csv --out data/raw/BTCUSDT-1min.parquet

run:
	backtest run --config config/strategy.yaml --bars data/raw/BTCUSDT-1min.parquet --out outputs

report:
	backtest report --config config/strategy.yaml --equity outputs/equity.parquet --out outputs

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
