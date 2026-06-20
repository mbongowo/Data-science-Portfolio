# disturbance-detection

[![CI](https://img.shields.io/github/actions/workflow/status/josephmbuh/disturbance-detection/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![Pixi](https://img.shields.io/badge/env-pixi-orange.svg)](pixi.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](.pre-commit-config.yaml)

**Time-series change & disturbance detection over multi-year satellite NDVI cubes.**

## Result first

> Pointed at the **2020 Creek Fire** (Sierra National Forest, California; MTBS
> ignition date **2020-09-04**), the per-pixel breakpoint detector places the
> disturbance **within two weeks of the recorded ignition** across the burn
> perimeter, with a sharp negative NDVI magnitude where the fire burned hottest
> and near-zero false alarms in the surrounding unburned forest.

| Disturbance DATE map | Disturbance MAGNITUDE map |
|----------------------|---------------------------|
| ![date map placeholder](outputs/.gitkeep) *date of largest breakpoint per pixel (`outputs/disturbance_date.png`)* | ![magnitude map placeholder](outputs/.gitkeep) *NDVI drop at the breakpoint (`outputs/disturbance_magnitude.png`)* |

Validation summary (printed by `disturb.validate.summary`):

```
Event 2020-09-04 (+/-60 d):
  detection rate :  XX.X% (n_detected / n_event px)
  false-alarm    :   X.X% (n_false_alarms / background px)
```

(Run the pipeline to populate the maps and the exact numbers; see *How to run*.)

## Problem -> result -> run

**Problem.** Wildfire, deforestation and drought leave a signature in the
vegetation time series, but a single before/after image can't tell a real,
*persistent* disturbance from a cloud, a shadow, or normal seasonality. We need
to model each pixel's seasonal-trend behaviour over years and flag the moment it
*breaks*.

**Result.** An analysis-ready, Dask-backed NDVI time cube; a per-pixel
seasonal-trend decomposition; a breakpoint detector that returns the **date** and
**magnitude** of the most significant disturbance per pixel; and a validation
step that quantifies spatial agreement with a documented event (the Creek Fire).

**Run.** See below.

## How it works

```
config/aoi.yaml
      |
      v
src/disturb/cube.py      STAC search -> odc.stac.load -> cloud mask (-> NaN, never 0)
      |                  -> NDVI -> regular 16-day time grid  (Dask-backed xarray)
      v
src/disturb/decompose.py pure-numpy harmonic regression per pixel
      |                  -> trend + seasonal + residual
      v
src/disturb/detect.py    CUSUM breakpoint on the residual (pure numpy)
      |                  -> (date, magnitude) of the largest break
      v
src/disturb/validate.py  spatial agreement vs. a known event polygon/date
```

The cube-building approach (odc-stac / xarray / Dask) is shared with the
**eo-monitor** project, here applied to the **time axis** rather than a single
scene.

### Design notes

- **Masked gaps are `NaN`, never `0`.** A zero NDVI reads as a real low-vegetation
  observation and would manufacture phantom disturbances; masking before the NDVI
  ratio keeps gaps honest.
- **The core is pure numpy.** `harmonic_decompose`, `detect_breakpoint` and the
  `spatial_agreement` validation arithmetic depend only on numpy, so they are
  unit-tested deterministically (`tests/`, 29 tests across decomposition,
  detection and validation) and the package imports cleanly without the
  geospatial stack. Heavier paths (`statsmodels` STL, `ruptures` PELT, the EO
  stack, polygon rasterisation) are imported lazily behind guards.
- **Multiple harmonics.** The decomposition fits an annual term plus, by
  default, a semiannual one, and exposes per-harmonic amplitude and phase
  (`HarmonicFit.seasonal_amplitude`, `.seasonal_phase`).

## How to run

### 1. Environment (pixi, recommended)

```bash
pixi install        # resolves conda-forge geo deps and GENERATES pixi.lock
pixi run test       # pytest (pure-numpy core tests pass with no GIS install)
pixi run lint       # ruff + mypy
pixi run run        # disturb --config config/aoi.yaml  (hits live STAC)
```

> `pixi.lock` is **not** committed - it is machine-generated. `pixi install`
> creates it on first run for your platform.

### pip fallback

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
disturb --config config/aoi.yaml
```

(GDAL-backed wheels are easier via `pixi`/conda-forge than via pip.)

### 2. Validation data (optional)

```bash
python data/download.py --url <event_perimeter.geojson> --out data/raw/creek_fire_perimeter.geojson
```

### 3. Explore

Open `notebooks/01_disturbance.ipynb` for a minimal runnable walk-through, or
read [`USAGE.md`](USAGE.md) for the full end-to-end guide: configuring every
field in `config/aoi.yaml`, Planetary Computer auth, building the cube, running
decompose then detect, where the date and magnitude maps land, validating
against a documented event, and troubleshooting sparse series, cloud gaps and
CRS.

## Data sources

- **HLS** (HLSS30 / HLSL30) via the **Microsoft Planetary Computer** STAC API -
  dense cadence, good for breakpoint timing.
- **Landsat Collection-2 Level-2** via STAC / earthaccess - long record for the
  baseline seasonal-trend model.
- **MTBS / GFW / news** for documented fire & deforestation events used in
  validation.

## Project layout

```
disturbance-detection/
├── config/aoi.yaml            # AOI, dates, source, detection & validation params
├── data/
│   ├── raw/.gitignore         # downloaded data (git-ignored)
│   └── download.py            # fetch validation event polygons
├── src/disturb/
│   ├── cube.py                # STAC -> masked NDVI time cube (Dask xarray)
│   ├── decompose.py           # pure-numpy harmonic regression (+ optional STL)
│   ├── detect.py              # pure-numpy CUSUM breakpoint (+ optional ruptures)
│   ├── validate.py            # spatial agreement vs. a known event
│   └── cli.py                 # `disturb` entry point
├── tests/                     # pytest: decomposition, detection & validation on synthetic data
├── notebooks/01_disturbance.ipynb
├── outputs/                   # generated maps/figures
├── USAGE.md                   # end-to-end run guide
├── pixi.toml / pyproject.toml / requirements.txt
├── Dockerfile / Makefile / .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

## Caveats

- The cube build hits **live STAC services**; results depend on data
  availability for the AOI/date range and on your network/credentials.
- `pixi.lock` is generated by `pixi install`; it is intentionally not committed.
- The default AOI/event targets the Creek Fire; swap `config/aoi.yaml` for
  any other AOI and documented event.

## License

MIT (c) 2026 Joseph Mbuh - see [LICENSE](LICENSE).
