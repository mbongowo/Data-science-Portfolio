# Convenience targets. These wrap pixi tasks where possible; plain-python
# fallbacks are provided for environments without pixi.
.PHONY: help install lint test run train smoke prepare clean

help:
	@echo "Targets:"
	@echo "  install   pixi install (resolves env + writes pixi.lock)"
	@echo "  lint      ruff check + mypy"
	@echo "  test      pytest"
	@echo "  prepare   generate tiny synthetic dataset"
	@echo "  smoke     1-step fast_dev_run training smoke"
	@echo "  train     full training run (needs torch + data + ideally a GPU)"
	@echo "  run       alias for train"

install:
	pixi install

lint:
	ruff check . && mypy src

test:
	pytest -q

prepare:
	python scripts/prepare_data.py --synthetic --out data/spacenet

smoke:
	python -m geoseg.train trainer.fast_dev_run=true data.batch_size=2 data.num_workers=0

train:
	python -m geoseg.train

run: train

clean:
	rm -rf outputs mlruns .pytest_cache .ruff_cache .mypy_cache
