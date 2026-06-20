# Model Card — geoai-segmentation building-footprint U-Net

This card follows the spirit of Mitchell et al. (2019), "Model Cards for Model
Reporting." Numbers marked *(placeholder)* must be filled in from a real run's
`metrics.json` — they are not fabricated here.

## Model details

- **Architecture:** U-Net (`segmentation_models_pytorch`) with a ResNet-34
  encoder (ImageNet-pretrained), single-channel logit head (binary footprint).
- **Loss:** equally weighted Dice + BCE-with-logits.
- **Framework:** PyTorch + PyTorch Lightning.
- **Inputs:** 3-band (RGB) GeoTIFF tiles, default 256×256, per-band standardised.
- **Outputs:** per-pixel building probability → thresholded binary mask, written
  back as a **georeferenced** GeoTIFF (CRS/transform copied from the input).
- **Version:** 0.1.0. **License:** MIT. **Author:** Joseph Mbuh.

## Intended use

- **Primary:** rapid building-footprint extraction from RGB EO imagery for
  mapping, population/exposure estimation, and disaster-response basemaps.
- **Users:** GIS analysts, humanitarian mapping teams, EO/ML researchers.
- **Out of scope:** instance-level building counting/delineation (this is
  *semantic* segmentation, not instance segmentation); cadastral or legal
  boundary determination; safety-critical decisions without human review.

## Training data

- **Source:** SpaceNet building footprints (AWS Open Data); optionally Google
  Open Buildings or torchgeo benchmark loaders. See `scripts/prepare_data.py`.
- **Preparation:** imagery tiled to fixed-size windows; vector footprints
  rasterised to aligned binary masks; per-band normalisation.
- **Splits:** deterministic hash-based train/val/test (70/15/15), stable across
  machines and seeds (`geoseg.datamodule.deterministic_split`).

## Evaluation

- **Metrics:** IoU (Jaccard) and F1/Dice on the held-out **test** split,
  computed with the pure-numpy `geoseg.metrics` module. The same module also
  provides precision, recall, and (for multi-class label maps) per-class and
  macro IoU, all with `ignore_index` support and a defined empty-mask
  convention. `geoseg.evaluate` reports IoU and F1 by default.
- **Results:** IoU `0.78` *(placeholder)*, F1 `0.86` *(placeholder)*. These two
  numbers are illustrative only and were not produced by a training run in this
  repository. Replace them with the values from your own `metrics.json` after a
  real run; do not cite them as measured performance.
- **How to regenerate:** run `geoseg.evaluate` against a checkpoint and the test
  split (writes `metrics.json` plus a qualitative `panel.png`). The metric code
  is unit-tested against hand-computed values, so a given (prediction, target)
  pair yields the same score on any machine.

## Quantitative provenance

Every training run logs to MLflow: the fully **resolved** Hydra config and the
**git commit SHA**, so any reported number is traceable to exact code + settings.

## Ethical considerations & limitations

- **Geographic bias:** SpaceNet covers specific cities; performance degrades on
  unseen regions, building styles (e.g. informal settlements, rural structures),
  and roofing materials under-represented in training.
- **Sensor / resolution shift:** trained on particular GSD and RGB sensors;
  different resolution, off-nadir angle, or spectral response hurts accuracy.
- **Seasonal / atmospheric effects:** shadows, snow, haze, and seasonal
  vegetation can cause false positives/negatives.
- **Class imbalance:** sparse-building tiles dominate the empty class; the Dice
  term mitigates but does not eliminate this.
- **Dual-use / privacy:** building footprints can be sensitive; downstream use
  for surveillance or targeting is explicitly discouraged.

## Failure modes

- Confuses large impervious surfaces (parking lots, courtyards) with rooftops.
- Merges adjacent buildings (no instance separation).
- Misses very small or partially occluded structures.

## Maintenance

Re-train and re-evaluate when applying to a new region or sensor. Recompute
per-band normalisation statistics on the target dataset.
