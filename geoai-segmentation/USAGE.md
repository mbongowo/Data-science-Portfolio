# Usage

This guide walks through the repository from a clean checkout to a reproduced
number. Every run records the seed, the resolved config, and the git commit, so
a result can be traced back to the exact code and settings that produced it.

If you only want to check the parts that do not need a model, jump to
[Run the numpy-only tests](#run-the-numpy-only-tests). Those tests cover the
metric formulas, tiling, splitting, normalisation, and seeding, and they run on
a machine with nothing but numpy installed. For a single-command demonstration
with real numbers, see [Run the demo](#run-the-demo) — no GPU, no data, under a
second.

## Run the demo

`geoseg.demo` is a self-contained, GPU-free demonstration of the tested
pure-numpy core. It synthesises a small seeded 4-class label map and a
deliberately noisy prediction, drives the real metric functions over them, and
writes artifacts:

```bash
pixi run demo            # or: python -m geoseg.demo  (or: make demo)
```

This prints a summary dict and writes three files to `outputs/`:

- `per_class_iou.csv` — IoU for each class (blank for absent classes).
- `confusion.csv` — the `N×N` confusion matrix (rows = true, cols = predicted).
- `summary.json` — mean IoU, pixel accuracy, frequency-weighted IoU, Cohen's
  kappa, and binary foreground IoU/F1/precision/recall.

The numbers are deterministic for a given seed (default 0) and are pinned in
`tests/test_demo.py`. They come from small seeded synthetic masks, so they
exercise the metric code honestly but are not a model result. Override the seed
with `python -m geoseg.demo --seed 7`.

`notebooks/01_walkthrough.ipynb` runs the same demo and additionally shows the
confusion matrix and a tiling round-trip with brief commentary.

## Install

The pinned path uses [pixi](https://pixi.sh), which resolves a conda-forge +
PyTorch environment and writes `pixi.lock` on first run:

```bash
pixi install
```

`pixi.lock` is not committed before resolution. After `pixi install` you can
commit it to pin the exact dependency graph for later runs.

If you do not use pixi, install in editable mode with pip:

```bash
python -m pip install -e .
```

That gives you the `geoseg-train` and `geoseg-infer` console scripts and the
`python -m geoseg.*` entry points.

### GPU notes

Training needs PyTorch with CUDA and a GPU; the model and the data loaders are
the only parts that require it. The metric and data-prep logic is pure-numpy and
runs on CPU.

- The Trainer is configured with `accelerator: auto` and `devices: auto`
  (`conf/config.yaml`), so it uses a GPU when one is visible and falls back to
  CPU otherwise. CPU training is only practical for the 1-step smoke run.
- Install a CUDA-matched PyTorch build. With pixi this comes from the locked
  environment. With pip, follow the selector at pytorch.org for your CUDA
  version.
- For bit-for-bit reproducibility the seed utility requests deterministic
  algorithms and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Determinism on GPU is
  best-effort: some cuDNN kernels have no deterministic variant, and results can
  still differ across GPU models or driver versions.

## Get data

`scripts/prepare_data.py` has three modes.

Synthetic data, for exercising the plumbing without any download (needs rasterio
+ numpy):

```bash
python scripts/prepare_data.py --synthetic --out data/spacenet --n 8
```

This fabricates small GeoTIFF image/mask pairs with random rectangular
"buildings". It is enough to run the smoke train and confirm the graph wires up,
but it is not real data and any metric on it is meaningless.

Real SpaceNet building footprints live in the AWS Open Data registry and are
free to read with `--no-sign-request`:

```bash
aws s3 ls s3://spacenet-dataset/ --no-sign-request
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/ ./data/raw/ \
    --recursive --no-sign-request
```

After download, rasterise the GeoJSON footprints to masks aligned to each image
tile (`rasterio.features.rasterize`) and place the results so that filenames
match:

```
data/spacenet/images/<name>.tif
data/spacenet/masks/<name>.tif
```

Google Open Buildings (CC-BY-4.0) is an alternative vector source covering the
Global South. `python scripts/prepare_data.py --torchgeo` prints how to use a
torchgeo benchmark loader instead of raw SpaceNet.

## The Hydra config system

Configuration is composed from `conf/`:

- `conf/config.yaml` is the root. Its `defaults` list pulls in `model: unet` and
  `data: spacenet`, then `_self_` so the root file's own keys win. It also sets
  the seed, the MLflow tracking URI, and the Trainer block.
- `conf/model/unet.yaml` holds the encoder, channel counts, learning rate, and
  loss weights.
- `conf/data/spacenet.yaml` holds the data directory, tile size, stride, batch
  size, per-band normalisation stats, and split fractions.

Every knob is a tracked YAML value, and any of them can be overridden on the
command line with dotted paths:

```bash
# swap the encoder and double the batch size
python -m geoseg.train model.encoder_name=resnet50 data.batch_size=16

# change the seed for a replicate run
python -m geoseg.train seed=7

# point at a different dataset directory
python -m geoseg.train data.data_dir=./data/my_region
```

Hydra writes each run into a timestamped directory under `paths.output_dir`
(`outputs/<date_time>/`), so successive runs do not overwrite each other.

## Train

```bash
# 1-step smoke run: verifies the model, data, and trainer wire up
python -m geoseg.train trainer.fast_dev_run=true

# full run
python -m geoseg.train
```

Under pixi the same commands are available as `pixi run smoke` and
`pixi run train`.

### How MLflow logging and git SHA capture work

`geoseg.train` logs to MLflow at the URI in `mlflow_tracking_uri` (default
`file:./mlruns`, a local directory). Two things make a run reproducible after the
fact:

- The **resolved** config is logged as `resolved_config.yaml`. Resolved means
  Hydra has already merged the defaults and applied your CLI overrides and any
  interpolations, so the artifact is the exact configuration the run used, not
  the templates.
- The **git commit SHA** is logged as a parameter via `geoseg.train.git_sha`,
  which shells out to `git rev-parse HEAD`. If git is unavailable the value is
  recorded as `unknown` rather than failing the run. Commit before training so
  this points at real code.

Browse runs with `mlflow ui --backend-store-uri ./mlruns`.

## Evaluate

Evaluation runs the trained checkpoint over the held-out test split and writes
two artifacts:

- `metrics.json`: mean IoU and F1 over the test tiles, computed by the
  pure-numpy `geoseg.metrics`. The aggregation in `geoseg.evaluate.aggregate_metrics`
  is unit-tested, so the reported numbers are stable for a given set of
  predictions and targets.
- `panel.png`: a qualitative `image | ground truth | prediction` panel for a few
  test tiles.

```python
from geoseg.evaluate import evaluate

evaluate(
    checkpoint="checkpoints/best.ckpt",
    data_dir="data/spacenet",
    output_dir="outputs/eval",
    tile_size=256,
    threshold=0.5,
)
```

The metric module exposes more than the two headline numbers: per-class IoU and
macro IoU for multi-class label maps, a dense confusion matrix, pixel accuracy,
frequency-weighted IoU, Cohen's kappa, and per-class precision/recall, plus the
binary precision and recall — all with `ignore_index` support for void pixels and
a defined convention for empty masks (an empty prediction that matches an empty
target scores 1.0). These run without torch and can be called directly on numpy
arrays; `geoseg.demo` exercises them end to end.

## Infer on a new tile

The inference CLI predicts a mask for one GeoTIFF and writes a **georeferenced**
GeoTIFF whose CRS and transform are copied from the input, so the output mask
lines up with the source raster in any GIS.

```bash
python -m geoseg.infer run \
    --checkpoint checkpoints/best.ckpt \
    --input data/new_tile.tif \
    --output outputs/new_tile_mask.tif \
    --threshold 0.5
```

The output is a single-band uint8 mask ({0,1}) with LZW compression and `nodata`
set to 0. The spatial profile is taken verbatim from the input, so no
reprojection happens here; feed it a tile already in the CRS you want.

## Reproduce exact numbers

To reproduce a result, pin the three things a run depends on:

1. **Seed.** Set `seed=<n>` (the default is 42). `geoseg.seed.seed_everything`
   seeds python's `random`, numpy, and torch, and requests deterministic
   algorithms. The numpy and python RNG paths are deterministic on any machine;
   the GPU path is best-effort as noted above.
2. **Config.** Use the same `conf/` values and the same CLI overrides. The
   logged `resolved_config.yaml` from the original run is the authoritative
   record; diff against it if a rerun disagrees.
3. **Checkpoint and code.** Evaluate the same `.ckpt` and check out the git SHA
   that the run logged to MLflow. Same code plus same checkpoint plus same data
   gives the same `metrics.json`.

The deterministic split (`geoseg.datamodule.deterministic_split`) assigns each
tile to train/val/test by hashing the tile key with the seed, so the split is
stable across machines and does not depend on file ordering or on holding the
dataset in memory. A given key and seed always land in the same split.

## Run the numpy-only tests

These need only numpy and confirm the formulas and helpers behave as documented:

```bash
PYTHONPATH=src python -m pytest tests/
```

They cover the metrics (IoU, F1, precision, recall, per-class and macro IoU,
confusion matrix, pixel accuracy, frequency-weighted IoU, Cohen's kappa,
`ignore_index`, empty-mask handling), the sliding-window tiling (`tile_indices`
coverage and `stitch` round-trip with and without overlap), the demo (pinned
seed-0 numbers), the tiling grid in the datamodule including partial and
overlapping tiles, the deterministic split (stability, proportions, no overlap,
order preservation), per-band normalisation, and the seed utility. Full training
and the torch-dependent inference and evaluation paths additionally need
PyTorch, rasterio, and a checkpoint, and full metrics on real data need the
SpaceNet download.
