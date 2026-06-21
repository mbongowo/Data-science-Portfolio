# Data Science Portfolio

Spatial data science work by **Joseph Mbuh**: remote sensing, Python geospatial pipelines, GeoAI, and analysis that runs the same way on someone else's machine as it does on mine.

Six self-contained projects live here, each in its own folder with its own environment, tests, CI, and documentation. They are ordered so the skill areas alternate, and so the later ones reuse code from the earlier ones. The deployed app imports the flagship pipeline as a dependency, and the change-detection project builds its data cube the same way the flagship does, so the repositories visibly compose rather than repeat.

## The projects

| # | Folder | What it does | Main skill |
|---|--------|--------------|------------|
| 1 | [`eo-monitor`](./eo-monitor) | Pulls Sentinel-2 from a STAC catalogue over an area of interest, computes vegetation and moisture indices, scores anomalies against a baseline, and writes cloud-optimised GeoTIFFs. One command, no manual downloads. | Cloud-native EO pipeline (STAC, Dask, COGs) |
| 2 | [`access-to-care`](./access-to-care) | Travel time from each populated place to the nearest health facility over a road network, weighted by population to show who is far from care. Built around Cameroon. | Vector and network analysis |
| 3 | [`spatial-hotspots`](./spatial-hotspots) | Exploratory spatial data analysis: spatial weights, global and local autocorrelation, cluster and outlier maps, with the interpretation written out and the limits stated. | Spatial statistics and significance |
| 4 | [`geoai-segmentation`](./geoai-segmentation) | A semantic-segmentation model on Earth-observation imagery, set up so a reported number can be reproduced from the committed seed, config, and checkpoint. | Reproducible deep learning + model card |
| 5 | [`disturbance-detection`](./disturbance-detection) | A multi-year NDVI time cube, a per-pixel seasonal-trend fit, breakpoint detection, and maps of the date and size of disturbance, checked against a recorded event. | Time-series change detection |
| 6 | [`eo-explorer-app`](./eo-explorer-app) | A web app where you draw an area, pick a date and an index, and see live Sentinel-2 rendered on a map. The index code is imported from `eo-monitor`. | A deployable interactive app |

## How they fit together

`eo-explorer-app` depends on `eo-monitor` and calls its index functions directly, so the app and the pipeline share one definition of NDVI rather than two copies. `disturbance-detection` reuses the same STAC-to-xarray cube pattern as `eo-monitor`, pointed at the time axis instead of a single date. The other three stand alone.

Build and reading order: 1, 2, 3, 5, 4, 6. If you only have time for three, projects 1, 3, and 6 already cover a cloud-native pipeline, real spatial statistics, and a shipped app.

## Running a project

**New here? Read [RUNNING.md](./RUNNING.md)** — it has copy-paste steps for each of
the six projects in one place, including which ones need a (free) API key.

The short version: every folder has its own `README.md` and `USAGE.md`, and from a
clean clone the pattern is the same for all of them:

```bash
cd <project>
pixi install            # generates the lockfile and the environment
pixi run test           # quick check; needs no network or API keys
```

`pixi` uses conda-forge, which is the reliable way to install GDAL and the rest of the geospatial stack. A `pip` path is provided as a fallback for the parts that do not need compiled geo libraries.

## What runs today, and what needs data or compute

The pure-numerical core of each project (index math, equity statistics, Moran's I and the local cluster statistics, segmentation metrics and tiling, harmonic decomposition and breakpoint detection, the app's geometry and caching helpers) has a real test suite that passes with only `numpy` and `pytest` installed. That is what the continuous integration runs on every push.

The full pipelines need the geospatial environment and live data. Pulling imagery from STAC, training the segmentation model on a GPU, and serving the app are documented in each project's `USAGE.md` but are not exercised in CI, because they need network access, large downloads, or hardware that a CI runner does not have. The lockfiles are not committed; `pixi install` generates them per platform on first run.

## Conventions shared across the projects

Each repository uses a `src/` layout with an importable package, a `pyproject.toml` with a console entry point, `pixi.toml` and `pyproject.toml` for the environment, a `Dockerfile`, a `Makefile` with `run` / `test` / `lint`, ruff and mypy through pre-commit, pytest with small committed fixtures, and config in YAML rather than hard-coded in source. Data directories are git-ignored and fetched by a script. Keeping these identical across six repos is deliberate.

## License

MIT, on each project. See the `LICENSE` file inside each folder.
