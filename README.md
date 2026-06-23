# Data Science Portfolio

Data science work by **Joseph Mbuh**: spatial / remote-sensing pipelines and
large-scale (big data) analytics, built so the same code runs the same way on
someone else's machine as it does on mine.

Each project lives in its own folder with its own environment, tests, CI, and
documentation. The work splits into a **spatial** track (remote sensing,
geospatial pipelines, GeoAI), a **big data** track (streaming, distributed
processing, the modern data stack), and a set of **technique-replication**
projects that rebuild the idea behind a well-known open-source repo on a new
dataset or region. Every folder stands alone; pick by the role you are reading
for. Across all of them the pure-numerical core runs and is tested in CI, while
the heavy parts (GPU, Earth Engine, Docker/Kafka, Terraform/Azure, deploys) are
documented to run on your own machine.

## Spatial track

Six projects, ordered so the skill areas alternate and the later ones reuse code
from the earlier ones. The deployed app imports the flagship pipeline as a
dependency, and the change-detection project builds its data cube the same way
the flagship does, so the projects visibly compose rather than repeat.

| # | Folder | What it does | Main skill |
|---|--------|--------------|------------|
| 1 | [`eo-monitor`](./spatial/eo-monitor) | Pulls Sentinel-2 from a STAC catalogue over an area of interest, computes vegetation and moisture indices, scores anomalies against a baseline, and writes cloud-optimised GeoTIFFs. One command, no manual downloads. | Cloud-native EO pipeline (STAC, Dask, COGs) |
| 2 | [`access-to-care`](./spatial/access-to-care) | Travel time from each populated place to the nearest health facility over a road network, weighted by population to show who is far from care. Built around Cameroon. | Vector and network analysis |
| 3 | [`spatial-hotspots`](./spatial/spatial-hotspots) | Exploratory spatial data analysis: spatial weights, global and local autocorrelation, cluster and outlier maps, with the interpretation written out and the limits stated. | Spatial statistics and significance |
| 4 | [`geoai-segmentation`](./spatial/geoai-segmentation) | A semantic-segmentation model on Earth-observation imagery, set up so a reported number can be reproduced from the committed seed, config, and checkpoint. | Reproducible deep learning + model card |
| 5 | [`disturbance-detection`](./spatial/disturbance-detection) | A multi-year NDVI time cube, a per-pixel seasonal-trend fit, breakpoint detection, and maps of the date and size of disturbance, checked against a recorded event. | Time-series change detection |
| 6 | [`eo-explorer-app`](./spatial/eo-explorer-app) | A web app where you draw an area, pick a date and an index, and see live Sentinel-2 rendered on a map. The index code is imported from `eo-monitor`. **[▶ Live demo](https://data-science-portfolio-kpnqhpxmfzgwpgwbxkejql.streamlit.app/)** | A deployable interactive app |

`eo-explorer-app` depends on `eo-monitor` and calls its index functions directly,
so the app and the pipeline share one definition of NDVI rather than two copies.
`disturbance-detection` reuses the same STAC-to-xarray cube pattern as
`eo-monitor`, pointed at the time axis instead of a single date. Build and reading
order: 1, 2, 3, 5, 4, 6. If you only have time for three, projects 1, 3, and 6
already cover a cloud-native pipeline, real spatial statistics, and a shipped app.

Like the big-data track, each of these now ships a one-command demo
(`pixi run demo` / `make demo`) that drives the real pure-numpy core over a small
seeded synthetic input in seconds, so the numbers in each README are reproducible,
and a walkthrough notebook in `notebooks/`.

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
| 1 | [`clickstream-pipeline`](./non-spatial/clickstream-pipeline) | A real-time event-analytics pipeline: ingest a high-volume stream and turn raw clicks into live windowed metrics, with a watermarking story for late/out-of-order events. | Kafka + Spark Structured Streaming |
| 2 | [`log-anomaly`](./non-spatial/log-anomaly) | Parse tens of millions of messy machine logs into templates and event-count features, then flag anomalies and score precision/recall against the labelled HDFS set. | Spark + scikit-learn |
| 3 | [`als-recommender`](./non-spatial/als-recommender) | Collaborative filtering at scale on MovieLens-25M, evaluated like a real ML system: RMSE plus ranking metrics against an honest popularity baseline. | Spark MLlib (ALS) |
| 4 | [`sentiment-scale`](./non-spatial/sentiment-scale) | Process tens of millions of Reddit posts into a sentiment-over-time dataset, with the lexicon scoring validated on a labelled sample. | Spark + NLP |
| 5 | [`tlc-analytics`](./non-spatial/tlc-analytics) | An honest engine bake-off on billions of NYC taxi rows (no geo): the same analytical workload across Spark, DuckDB, and a warehouse, benchmarked. | Spark / DuckDB / warehouse |
| 6 | [`dbt-modern-stack`](./non-spatial/dbt-modern-stack) | The pipeline employers actually hire for: batch ETL into a warehouse, layered dbt models with tests and lineage, orchestration, and a BI layer. | dbt + DuckDB + orchestration |
| 7 | [`crypto-backtest`](./non-spatial/crypto-backtest) | High-frequency tick data resampled to bars and run through a reproducible backtest — emphasis on rigour: realistic fees/slippage and no look-ahead bias. | Spark / Polars |
| 8 | [`graph-analysis`](./non-spatial/graph-analysis) | Graph algorithms at scale on a large SNAP network: PageRank, community detection, and triangle counting, with the results actually interpreted. | Spark GraphFrames |

Depth beats breadth: each of these is a complete, documented, reproducible piece
rather than a stub. For a data-engineering read, start with `clickstream-pipeline`,
`tlc-analytics`, and `dbt-modern-stack`.

## Technique-replication projects

Ten further projects — five spatial, five non-spatial — that replicate the
technique behind a well-known open-source repository on a new dataset or region
(mostly **Cameroon**) and add what the original lacks: hand-derived tests, a
deployable app, an Azure path, or honest validation against an official source.
Each rebuilds the idea rather than cloning it, with a pure-numpy/pandas core that
runs and is tested in CI and a default area-of-interest you can swap for your own.

### Spatial (`spatial/01–05`)

| Folder | What it shows | Inspired by |
|--------|---------------|-------------|
| [`01-segment-geospatial`](./spatial/01-segment-geospatial) | Segment-Anything masks → counted/measured buildings & fields over Douala | [opengeos/segment-geospatial](https://github.com/opengeos/segment-geospatial) |
| [`02-earth-engine-timeseries`](./spatial/02-earth-engine-timeseries) | Multi-year Sentinel-2 change / forest-loss; STAC-default (auth-free), geemap optional | [giswqs/geemap](https://github.com/giswqs/geemap) |
| [`03-torchgeo-landcover`](./spatial/03-torchgeo-landcover) | Land-cover classification, pretrained-vs-from-scratch, with a model card | [microsoft/torchgeo](https://github.com/microsoft/torchgeo) |
| [`04-leafmap-dashboard`](./spatial/04-leafmap-dashboard) | Deployable clinic-access dashboard (who is farthest from a clinic?) **[▶ Live](https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/)** | [opengeos/leafmap](https://github.com/opengeos/leafmap) |
| [`05-change-detection`](./spatial/05-change-detection) | SAR flood mapping (Otsu + before/after), validated against an OCHA report | [robmarkcole/satellite-image-deep-learning](https://github.com/robmarkcole/satellite-image-deep-learning) |

### Non-spatial (`non-spatial/01–05`)

| Folder | What it shows | Inspired by |
|--------|---------------|-------------|
| [`01-data-engineering-pipeline`](./non-spatial/01-data-engineering-pipeline) | Cameroon-weather ingest → warehouse → dbt → dashboard; DuckDB free or Azure | [DataTalksClub/data-engineering-zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) |
| [`02-mlops-pipeline`](./non-spatial/02-mlops-pipeline) | Rain-day model with MLflow tracking, FastAPI/Docker serving, PSI/KS drift monitoring | [DataTalksClub/mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) |
| [`03-streaming-pipeline`](./non-spatial/03-streaming-pipeline) | Real-time air-quality streaming with EPA AQI and threshold/spike alerting | [damklis/DataEngineeringProject](https://github.com/damklis/DataEngineeringProject) |
| [`04-ml-web-app`](./non-spatial/04-ml-web-app) | Deployable crop recommender (soil & climate → ranked crop) **[▶ Live](https://mbongowo-data-s-non-spatial04-ml-web-appappstreamlit-app-l5yxjk.streamlit.app/)** | [shsarv/Machine-Learning-Projects](https://github.com/shsarv/Machine-Learning-Projects) |
| [`05-focused-ml-project`](./non-spatial/05-focused-ml-project) | RAG question-answering over this portfolio's docs; free extractive, optional LLM | [ashishpatel26/500-…-Projects](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code) |

Several ship a deployable Streamlit app with an "Open in Streamlit" badge, and the
cloud-touching ones (data engineering, MLOps, streaming) carry both a free local
path and an opt-in **Azure** path.

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
continuous integration runs on every push, across all twenty-four projects.

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
