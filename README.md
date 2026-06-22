# Data Science Portfolio

Data science work by **Joseph Mbuh**: spatial / remote-sensing pipelines and
large-scale (big data) analytics, built so the same code runs the same way on
someone else's machine as it does on mine.

Each project lives in its own folder with its own environment, tests, CI, and
documentation. There are two tracks: a **spatial** track (remote sensing,
geospatial pipelines, GeoAI) and a **big data** track (streaming, distributed
processing, the modern data stack — no geospatial component). Pick by the role
you are reading for; every folder stands alone.

## Spatial track

Six projects, ordered so the skill areas alternate and the later ones reuse code
from the earlier ones. The deployed app imports the flagship pipeline as a
dependency, and the change-detection project builds its data cube the same way
the flagship does, so the projects visibly compose rather than repeat.

| # | Folder | What it does | Main skill |
|---|--------|--------------|------------|
| 1 | [`eo-monitor`](./eo-monitor) | Pulls Sentinel-2 from a STAC catalogue over an area of interest, computes vegetation and moisture indices, scores anomalies against a baseline, and writes cloud-optimised GeoTIFFs. One command, no manual downloads. | Cloud-native EO pipeline (STAC, Dask, COGs) |
| 2 | [`access-to-care`](./access-to-care) | Travel time from each populated place to the nearest health facility over a road network, weighted by population to show who is far from care. Built around Cameroon. | Vector and network analysis |
| 3 | [`spatial-hotspots`](./spatial-hotspots) | Exploratory spatial data analysis: spatial weights, global and local autocorrelation, cluster and outlier maps, with the interpretation written out and the limits stated. | Spatial statistics and significance |
| 4 | [`geoai-segmentation`](./geoai-segmentation) | A semantic-segmentation model on Earth-observation imagery, set up so a reported number can be reproduced from the committed seed, config, and checkpoint. | Reproducible deep learning + model card |
| 5 | [`disturbance-detection`](./disturbance-detection) | A multi-year NDVI time cube, a per-pixel seasonal-trend fit, breakpoint detection, and maps of the date and size of disturbance, checked against a recorded event. | Time-series change detection |
| 6 | [`eo-explorer-app`](./eo-explorer-app) | A web app where you draw an area, pick a date and an index, and see live Sentinel-2 rendered on a map. The index code is imported from `eo-monitor`. | A deployable interactive app |

`eo-explorer-app` depends on `eo-monitor` and calls its index functions directly,
so the app and the pipeline share one definition of NDVI rather than two copies.
`disturbance-detection` reuses the same STAC-to-xarray cube pattern as
`eo-monitor`, pointed at the time axis instead of a single date. Build and reading
order: 1, 2, 3, 5, 4, 6. If you only have time for three, projects 1, 3, and 6
already cover a cloud-native pipeline, real spatial statistics, and a shipped app.

## Big data track

Eight non-spatial projects, each scoped to stand alone on a free, genuinely large
public dataset, with an architecture that runs locally or on a free cloud tier.
The differentiator in each is honesty: a benchmark with numbers, an evaluation
metric against labels, or a quantified finding — not a tutorial.

Every one ships a one-command demo (`pixi run demo` or `make demo`) that drives
the real numeric core over a small, seeded synthetic dataset in seconds, so the
numbers in each README are reproducible rather than illustrative. A walkthrough
notebook in each `notebooks/` folder shows the core and the extra capabilities in
use.

| # | Folder | What it does | Core engine |
|---|--------|--------------|-------------|
| 1 | [`clickstream-pipeline`](./clickstream-pipeline) | A real-time event-analytics pipeline: ingest a high-volume stream and turn raw clicks into live windowed metrics, with a watermarking story for late/out-of-order events. | Kafka + Spark Structured Streaming |
| 2 | [`log-anomaly`](./log-anomaly) | Parse tens of millions of messy machine logs into templates and event-count features, then flag anomalies and score precision/recall against the labelled HDFS set. | Spark + scikit-learn |
| 3 | [`als-recommender`](./als-recommender) | Collaborative filtering at scale on MovieLens-25M, evaluated like a real ML system: RMSE plus ranking metrics against an honest popularity baseline. | Spark MLlib (ALS) |
| 4 | [`sentiment-scale`](./sentiment-scale) | Process tens of millions of Reddit posts into a sentiment-over-time dataset, with the lexicon scoring validated on a labelled sample. | Spark + NLP |
| 5 | [`tlc-analytics`](./tlc-analytics) | An honest engine bake-off on billions of NYC taxi rows (no geo): the same analytical workload across Spark, DuckDB, and a warehouse, benchmarked. | Spark / DuckDB / warehouse |
| 6 | [`dbt-modern-stack`](./dbt-modern-stack) | The pipeline employers actually hire for: batch ETL into a warehouse, layered dbt models with tests and lineage, orchestration, and a BI layer. | dbt + DuckDB + orchestration |
| 7 | [`crypto-backtest`](./crypto-backtest) | High-frequency tick data resampled to bars and run through a reproducible backtest — emphasis on rigour: realistic fees/slippage and no look-ahead bias. | Spark / Polars |
| 8 | [`graph-analysis`](./graph-analysis) | Graph algorithms at scale on a large SNAP network: PageRank, community detection, and triangle counting, with the results actually interpreted. | Spark GraphFrames |

Depth beats breadth: each of these is a complete, documented, reproducible piece
rather than a stub. For a data-engineering read, start with `clickstream-pipeline`,
`tlc-analytics`, and `dbt-modern-stack`.

## Running a project

**New here? Read [RUNNING.md](./RUNNING.md)** — it has copy-paste steps for the
spatial projects in one place, including which ones need a (free) API key.

The short version: every folder has its own `README.md` and `USAGE.md`, and from a
clean clone the pattern is the same for all of them:

```bash
cd <project>
pixi install            # generates the lockfile and the environment
pixi run test           # quick check; needs no network or API keys
```

`pixi` uses conda-forge, which is the reliable way to install the heavier stacks
(GDAL for the spatial projects; Spark/JVM-backed tools for the big-data ones). A
`pip` path is provided as a fallback for the parts that do not need compiled
dependencies.

## What runs today, and what needs data or compute

The pure-numerical core of every project has a real known-answer test suite that
passes with only `numpy`, `pandas`, and `pytest` installed. That is what the
continuous integration runs on every push, across all fourteen projects.

- **Spatial cores:** index math, equity statistics, Moran's I and the local
  cluster statistics, segmentation metrics and tiling, harmonic decomposition and
  breakpoint detection, the app's geometry and caching helpers.
- **Big-data cores:** window/sessionisation logic, log templating and PCA
  anomaly scoring, ALS factorisation and ranking metrics, lexicon scoring and
  TF-IDF, the taxi analytics marts, the dbt-style data-quality runner, OHLCV
  resampling and the no-look-ahead backtest engine, and PageRank / triangle /
  community algorithms.

The full pipelines need the heavier environment and live data. Pulling imagery
from STAC, training the segmentation model on a GPU, serving the app, running
Kafka and Spark, building a warehouse with dbt, and resampling multi-year tick
data are all documented in each project's `USAGE.md` but are not exercised in CI,
because they need network access, large downloads, a JVM/cluster, or hardware a
CI runner does not have. The heavy engines are kept behind lazily-imported wrapper
modules so the tested core never pulls them in. Lockfiles are not committed;
`pixi install` generates them per platform on first run.

## Conventions shared across the projects

Each project uses a `src/` layout with an importable package, a `pyproject.toml`
with a console entry point, `pixi.toml` and `pyproject.toml` for the environment,
a `Dockerfile`, a `Makefile` with `run` / `test` / `lint`, ruff and mypy through
pre-commit, pytest with hand-derived known-answer tests, and config in YAML rather
than hard-coded in source. Data directories are git-ignored and fetched by a
script. Keeping these identical across every folder is deliberate.

## License

MIT, on each project. See the `LICENSE` file inside each folder.
</content>
</invoke>
