# geoai-segmentation

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Lightning](https://img.shields.io/badge/PyTorch-Lightning-792ee5)](https://lightning.ai)
[![Reproducible](https://img.shields.io/badge/repro-pixi%20%2B%20hydra%20%2B%20mlflow-success)](pixi.toml)

**Semantic segmentation of building footprints from Earth-observation imagery, set up to reproduce.**

The point of this repo is not a leaderboard score. It is the engineering around the model: a pinned environment, config-driven runs, seeded experiments, logged metrics, a model card, and a single-command inference entry point. Clone it, run the documented commands with the same seed and config, and you get the same numbers. The metrics in `geoseg.metrics` are pure-numpy and tested, so the parts that decide what "the same number" means run without a GPU. See `USAGE.md` for the step-by-step path from install to reproduced metrics.

---

## Result first

A U-Net (ResNet-34 encoder, Dice + BCE loss) trained on SpaceNet building footprints.

| Split | IoU  | F1 (Dice) |
|-------|------|-----------|
| test  | `0.78` *(placeholder — fill from your `metrics.json`)* | `0.86` *(placeholder)* |

**Qualitative prediction panel** (`image | ground truth | prediction`):

![prediction panel](outputs/eval/panel.png)

> The panel and metrics above are produced by `geoseg.evaluate` and are committed
> nowhere by default (artifacts are git-ignored) — they regenerate deterministically.

---

## Why this is reproducible

| Concern            | How it's handled                                                        |
|--------------------|-------------------------------------------------------------------------|
| Environment        | `pixi.toml` (conda-forge + pytorch); `pixi install` writes `pixi.lock`   |
| Configuration      | Hydra (`conf/`) — every knob is a tracked YAML value                     |
| Randomness         | `seed.py` seeds python/numpy/torch + deterministic cuDNN flags           |
| Data splits        | Hash-based deterministic train/val/test (stable across machines)         |
| Experiment record  | MLflow logs the **resolved config** and the **git SHA** of every run     |
| Provenance         | Inference writes a **georeferenced** GeoTIFF (CRS/transform from input)  |

---

## Quickstart

```bash
# 1. Resolve the exact environment (creates pixi.lock on first run)
pixi install

# 2. Make a tiny synthetic dataset so the pipeline runs with no downloads
pixi run prepare-synthetic

# 3. One-step smoke train (verifies the whole graph wires up)
pixi run smoke

# 4. Full training run
pixi run train
#   override anything on the CLI, Hydra-style:
#   pixi run python -m geoseg.train model.encoder_name=resnet50 data.batch_size=16

# 5. Evaluate on the held-out test split (writes metrics.json + panel.png)
pixi run python -m geoseg.evaluate ... # see evaluate.py:evaluate()

# 6. Predict on a fresh tile -> georeferenced GeoTIFF mask
pixi run python -m geoseg.infer run \
    --checkpoint checkpoints/best.ckpt \
    --input data/new_tile.tif \
    --output outputs/new_tile_mask.tif
```

No pixi? Use pip (`pip install -e .`) and the `geoseg-train` / `geoseg-infer`
console scripts, or the `Makefile` targets (`make smoke`, `make train`, `make test`).

---

## Data

- **SpaceNet** building footprints (AWS Open Data, free / `--no-sign-request`).
- **Google Open Buildings** (CC-BY-4.0) as an alternative vector source.
- **torchgeo** built-in benchmark loaders for clean experimentation.

See `scripts/prepare_data.py` for download/prepare instructions and a
`--synthetic` mode that fabricates tiny GeoTIFFs for CI / smoke runs. SpaceNet
access:

```bash
aws s3 ls s3://spacenet-dataset/ --no-sign-request
```

Expected layout: `data/spacenet/images/*.tif` and `data/spacenet/masks/*.tif`
with matching filenames.

---

## Project layout

```
geoai-segmentation/
├── conf/                 # Hydra configs (config.yaml -> model/unet + data/spacenet)
├── src/geoseg/
│   ├── metrics.py        # pure-numpy IoU / F1 / precision / recall + multiclass IoU (no torch needed)
│   ├── seed.py           # seed-everything
│   ├── datamodule.py     # tiling, deterministic splits, augmentation, normalise
│   ├── model.py          # Lightning module: SMP U-Net + Dice/BCE, logs loss+IoU
│   ├── train.py          # Hydra entry; logs resolved config + git SHA to MLflow
│   ├── evaluate.py       # IoU/F1 on test split -> metrics.json + panel.png
│   └── infer.py          # typer CLI: predict mask, write georeferenced GeoTIFF
├── scripts/prepare_data.py
├── tests/                # pytest; metrics/datamodule/seed/evaluate tests pass with only numpy
└── .github/workflows/ci.yml
```

---

## Testing & quality

```bash
pixi run test     # pytest; the metrics/datamodule/seed/evaluate tests pass with numpy only
pixi run lint     # ruff + mypy
pre-commit install
```

The numpy-only tests cover the parts of the pipeline that do not need a model:
the metric formulas (IoU, F1, precision, recall, per-class and macro IoU, with
`ignore_index` and empty-mask handling), the tiling grid, the deterministic
split, per-band normalisation, and the seed utility. Run them on a bare machine
with `PYTHONPATH=src python -m pytest tests/`.

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest on Python 3.10 and
3.12 with a lightweight (torch-free) install, plus a best-effort CPU smoke-train
job that installs the full stack and runs a 1-step `fast_dev_run`.

---

## Caveats

- Full training needs **torch + a GPU** and the **real SpaceNet data**; the
  synthetic dataset only exercises the plumbing.
- `pixi.lock` is generated by `pixi install` (not committed pre-resolution).
- Headline metrics in the table are placeholders — regenerate them from your own
  `metrics.json` after a real run.

## License

MIT © 2026 Joseph Mbuh
