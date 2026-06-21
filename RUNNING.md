# How to run each project

This is the short, do-this version. Each project also has its own `README.md` and
`USAGE.md` with more detail, but you can run any of them with the steps here.

## Why pixi

Five of the six projects need GDAL and other geospatial C libraries. These are
painful to install with `pip` and easy with `pixi`, which installs them from
conda-forge. Install pixi once and every project becomes a two-line setup.

Install pixi:

- macOS / Linux: `curl -fsSL https://pixi.sh/install.sh | bash`
- Windows (PowerShell): `iwr -useb https://pixi.sh/install.ps1 | iex`

Close and reopen your terminal afterwards so `pixi` is on your PATH.

The pattern for every project is the same:

```bash
cd <project>      # e.g. cd eo-monitor
pixi install      # first time only: builds the environment and writes pixi.lock
pixi run <task>   # run something (tasks are listed per project below)
```

`pixi run test` works in every project and needs no network or API keys, so it
is the quickest way to confirm a project is set up correctly.

If you only want to read the code and run the unit tests without pixi, a plain
`pip install` in a virtual environment is enough for the tests; the full data
pipelines need the geospatial stack, which is why pixi is the recommended path.

---

## 1. eo-monitor — Sentinel-2 anomaly pipeline

No account or API key needed; it reads Sentinel-2 from the open Earth Search
catalogue.

```bash
cd eo-monitor
pixi install
pixi run run
```

`pixi run run` is `eo-monitor run --config config/corn_belt.yaml`. It searches for
imagery, computes NDVI/NDWI/NDMI and an anomaly against a baseline, and writes
Cloud-Optimised GeoTIFFs plus a preview PNG into `outputs/`. To change the area
or dates, edit `config/corn_belt.yaml`.

## 2. access-to-care — travel time to clinics

Needs the input data first. Most sources download without a key; the health
facility points come from Healthsites.io, which needs a free key.

```bash
cd access-to-care
pixi install

# Optional but recommended (free key from https://healthsites.io):
export HEALTHSITES_API_KEY=your_key          # Windows PowerShell: $env:HEALTHSITES_API_KEY="your_key"

pixi run download    # fetches OSM, facilities, population, and boundaries into data/raw
pixi run access      # computes travel time and the population-weighted equity stats
```

Outputs (an access-time map and a summary table) land in `outputs/`.

## 3. spatial-hotspots — cluster and hotspot statistics

Needs a free USDA NASS key for the crop-yield data.

```bash
cd spatial-hotspots
pixi install

# Free key from https://quickstats.nass.usda.gov/api:
export NASS_API_KEY=your_key                 # Windows PowerShell: $env:NASS_API_KEY="your_key"

pixi run download    # writes data/raw/iowa_corn_yield_2023.gpkg
pixi run analyze     # Moran's I, LISA, and Getis-Ord Gi*, with maps in outputs/
```

## 4. geoai-segmentation — segmentation model

The full training run needs a GPU and the real SpaceNet dataset. To check that
the pipeline works end to end without either, use the synthetic data and a
one-step run:

```bash
cd geoai-segmentation
pixi install
pixi run prepare-synthetic   # makes a tiny synthetic dataset in data/spacenet
pixi run smoke               # one training step on CPU, to prove the wiring works
```

For a real run on your own data and GPU: `pixi run train` (configured through the
files in `conf/`), then evaluate and predict as described in
`geoai-segmentation/USAGE.md`.

## 5. disturbance-detection — change over time

Reads from Microsoft Planetary Computer, which works anonymously (a subscription
key only raises rate limits, and is not required).

```bash
cd disturbance-detection
pixi install
pixi run run
```

`pixi run run` is `disturb --config config/aoi.yaml`. It builds an NDVI time
series, fits a seasonal model, detects the largest breakpoint per pixel, and
writes date-of-change and magnitude maps to `outputs/`. Edit `config/aoi.yaml` to
change the area, dates, or detection settings.

## 6. eo-explorer-app — the interactive web app

This one runs in your browser. It reuses the index code from `eo-monitor`, so
install that first.

```bash
cd eo-explorer-app
pixi install
pixi run install-eo-monitor    # installs the sibling eo-monitor package
pixi run app                   # opens the app at http://localhost:8501
```

In the app: draw a small box over land with the rectangle tool, pick a recent
date and an index, then click **Load imagery & compute index**. To put it online
instead, see the deployment steps in `eo-explorer-app/USAGE.md`.

---

## Running the tests for everything

From inside any project folder:

```bash
pixi run test
```

This runs that project's unit tests. They use small built-in fixtures, so they
need no network access and no API keys. The same tests run automatically in CI on
every push.
