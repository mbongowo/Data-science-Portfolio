# eo-monitor

[![CI](https://github.com/josephmbuh/eo-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/josephmbuh/eo-monitor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

## How badly did the 2023 flash drought stress the Corn Belt?

In mid-2023 a **rapid-onset ("flash") drought** developed across the western
U.S. Corn Belt and Northern Plains, stressing corn and soybean during the
critical July–August growing window. **eo-monitor** answers a concrete question:

> *Over a field-scale area of interest in eastern Nebraska, how far did
> peak-season vegetation greenness and canopy moisture in 2023 depart from the
> 2019–2022 normal?*

The pipeline runs from a clean clone with no manual downloads. Point it at a
config file and it discovers Sentinel-2 imagery from a public STAC catalogue,
masks clouds, computes NDVI / NDWI / NDMI, builds a per-pixel **z-score
anomaly** against a climatological baseline, and writes Cloud-Optimised GeoTIFFs
plus quicklook PNGs.

### Result

![NDVI anomaly](outputs/ndvi_anomaly.png)

*Generated artifact.* Running the pipeline writes `outputs/ndvi_anomaly.png`
(and `.tif`) along with the NDWI/NDMI equivalents. Negative (red) z-scores mark
pixels that were greener or wetter in past Julys/Augusts than in 2023. The image
is not committed (`outputs/` is git-ignored); it appears after you run the
command below.

## How to run

For a step-by-step guide (install, every config field, reading the anomaly map,
and troubleshooting) see [`USAGE.md`](USAGE.md).

The project ships a [pixi](https://pixi.sh) manifest so the entire geospatial
stack (GDAL, rasterio, odc-stac, …) resolves from conda-forge:

```bash
# 1. Install the environment (this also generates pixi.lock)
pixi install

# 2. Run the flagship config (2023 Corn Belt flash drought)
pixi run eo-monitor run --config config/corn_belt.yaml
```

Outputs land in `outputs/`:

```
outputs/
├── ndvi_anomaly.tif   ndwi_anomaly.tif   ndmi_anomaly.tif      # COGs
├── ndvi_composite.tif ndwi_composite.tif ndmi_composite.tif    # target medians
└── ndvi_anomaly.png   ndwi_anomaly.png   ndmi_anomaly.png       # quicklooks
```

Useful flags:

```bash
eo-monitor run --config config/corn_belt.yaml --max-items 30 --verbose
eo-monitor run -c config/corn_belt.yaml -o /tmp/run1
eo-monitor --help
```

### pip fallback

No pixi? A `requirements.txt` mirrors the runtime deps (GDAL/rasterio wheels
required):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
eo-monitor run --config config/corn_belt.yaml
```

> **Note:** `pixi.lock` is intentionally **not** committed. It is generated on
> your machine/CI by `pixi install`, pinning exact, platform-specific builds.

## What it does (pipeline)

```
config (YAML)
   └─> catalog.py   STAC search (Earth Search, sentinel-2-l2a), cloud + max-items filter
        └─> cube.py        odc-stac load -> lazy Dask xarray cube + SCL cloud mask
             └─> indices.py     NDVI / NDWI / NDMI (pure numpy/xarray)
                  └─> anomaly.py     per-pixel z = (value - baseline_mean) / baseline_std
                       └─> io.py          Cloud-Optimised GeoTIFF + PNG quicklook
```

### Indices

| Index | Formula | Bands (S2 L2A) | Sensitive to |
|-------|---------|----------------|--------------|
| NDVI  | (NIR − Red) / (NIR + Red)   | B08, B04 | green biomass / vigour |
| NDWI  | (Green − NIR) / (Green + NIR) | B03, B08 | open water / wetness |
| NDMI  | (NIR − SWIR) / (NIR + SWIR) | B08, B11 | canopy moisture |

### Configuration

Everything is config-driven (no hard-coded AOIs, dates, or thresholds). See
[`config/corn_belt.yaml`](config/corn_belt.yaml): AOI (bbox **or** vector path),
target date range, baseline window (+ climatological months), requested indices,
cloud-cover threshold, max-items guard, resolution/CRS and STAC endpoint.
Malformed config raises a clear `pydantic.ValidationError`.

## Tech stack

- **Discovery:** pystac-client, planetary-computer
- **Access:** odc-stac, rioxarray, xarray, dask (lazy; library code never calls `.compute()`)
- **Analysis:** numpy, xarray, spyndex
- **Output:** rio-cogeo, rasterio
- **CLI / config:** typer, pydantic-settings
- **Quality:** pytest, ruff, mypy, pre-commit, GitHub Actions
- **Reproducibility:** pixi (conda-forge) + Docker

## Development

```bash
make install   # pixi install (or pip install -e ".[dev]")
make test      # pytest
make lint      # ruff check + mypy
make run       # eo-monitor run --config config/corn_belt.yaml
```

The unit tests for `indices.py` and `anomaly.py` assert exact, hand-checked
values and run offline with only numpy (no network, no STAC, no GDAL).

## License

MIT © 2026 Joseph Mbuh
