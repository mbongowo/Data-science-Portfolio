# eo-monitor

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
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

The full Sentinel-2 pipeline needs the conda-forge geo stack and network access
to the STAC catalogue. For a result you can reproduce in under a second with
only numpy, the project ships an offline demo that drives the *same* index and
anomaly core over a small seeded synthetic cube with a planted vegetation-loss
patch:

```bash
pixi run demo          # or: make demo / eo-monitor demo
```

On `seed=0` (a 24×24 grid, planted 8×10 loss block = 80 pixels) it recovers the
planted patch exactly:

| metric | value |
|--------|-------|
| grid | 24 × 24 |
| mean target NDVI | 0.683 |
| anomaly pixels (\|z\| > 2) | 93 |
| anomaly fraction | 0.161 |
| max \|z\| | 42.12 |
| planted block recovered (recall) | 1.0 |

These are the actual numbers `run_demo(0)` prints; they are pinned in
`tests/test_demo.py`. The demo writes `outputs/summary.json` plus the NDVI and
z-score maps as `.npy`.

The real run writes `outputs/ndvi_anomaly.png` (and `.tif`) plus the NDWI/NDMI
equivalents; negative (red) z-scores mark pixels that were greener or wetter in
past Julys/Augusts than in 2023. `outputs/` is git-ignored.

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

`indices.py` ships a full single-scene spectral-index catalogue (34 indices),
all pure numpy/xarray and dispatchable by name via `compute_index` /
`required_bands` (`list_indices()` returns the full list). Grouped by category:

| Category | Indices |
|----------|---------|
| Vegetation | NDVI, EVI, EVI2, SAVI, MSAVI, GNDVI, ARVI, NDRE, VARI, RVI, DVI, CIgreen, CIrededge, MCARI, TCARI, LAI |
| Water & moisture | NDWI, MNDWI, NDMI, AWEI, NDII |
| Soil & geology | BSI, SI (salinity), IronOxide, ClayMinerals, FerrousMinerals |
| Built-up / urban | NDBI, UI, IBI |
| Snow / ice | NDSI, NDGI |
| Fire / burn | NBR, NBR2, BAI |

Sentinel-2 L2A assets used (Earth Search common names): `blue`, `green`, `red`,
`rededge1` (B05, red-edge / "RE"), `nir`, `swir16` (SWIR1, B11), `swir22`
(SWIR2, B12).

**Reflectance scaling.** Normalised-difference and pure-ratio indices are
scale-invariant. Indices with additive constants — **EVI, SAVI, MSAVI, AWEI,
BAI** (and LAI via EVI) — assume surface reflectance in `[0, 1]`; feed them
scaled reflectance, not raw DN.

**Caveats / equivalences.** `NDII` is the same formula as `NDMI` (kept under both
names). `NDSI` shares the formula of `MNDWI` (snow vs water is a threshold/context
call). `SI` is the `sqrt(Green*Red)` salinity form (several variants exist).
`NDGI` is the green/red glacier-index variant. `LAI = 3.618·EVI − 0.118` is an
**approximate empirical** relation, not a physical retrieval.

**Excluded by design** (not dispatchable): **EBBI** needs a thermal band, which
Sentinel-2 does not carry; **dNBR** needs two dates (pre/post fire) and so belongs
to the temporal/anomaly workflow (`anomaly.py`), not the single-scene index path.

### Capabilities

The pure-numpy core (`indices.py`, `anomaly.py`) is what the tests and the demo
exercise; it runs with only numpy installed.

- **Spectral indices:** the 34-index catalogue above, plus the public
  `normalized_difference(a, b)` building block. Each is dispatchable by name via
  `compute_index` / `required_bands`; `list_indices()` enumerates them.
- **Anomaly detection:** standard z-score (`anomaly_cube`, `zscore_anomaly`) and
  an outlier-resistant `robust_zscore` (median + MAD scaled by 1.4826).
- **Anomaly summaries:** `anomaly_fraction(z, threshold)` and
  `classify_anomaly(z, threshold)` (−1 loss / 0 none / +1 gain).
- Divide-by-zero, zero-variance, and NaN (masked-pixel) inputs all return NaN
  rather than raising; every helper has a hand-derived known-answer test.

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
make demo      # offline numpy demo -> outputs/summary.json (no geo stack)
make test      # pytest
make lint      # ruff check + mypy
make run       # eo-monitor run --config config/corn_belt.yaml
```

The unit tests for `indices.py` and `anomaly.py` assert exact, hand-checked
values and run offline with only numpy (no network, no STAC, no GDAL).

## License

MIT © 2026 Joseph Mbuh
