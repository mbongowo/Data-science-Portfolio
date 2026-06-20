# spatial-hotspots

[![CI](https://github.com/josephmbuh/spatial-hotspots/actions/workflows/ci.yml/badge.svg)](https://github.com/josephmbuh/spatial-hotspots/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Exploratory spatial data analysis (ESDA)**: build and compare spatial
weights matrices, test for global spatial autocorrelation, map local clusters
and spatial outliers (LISA / Getis-Ord Gi\*), and optionally fit a
Geographically Weighted Regression. Assumptions and interpretation are written
out, not hand-waved.

---

## Result first

**Question.** Does county-level **corn yield in Iowa (2023)** cluster in space,
or is it spatially random?

**Answer (illustrative).** Global Moran's I is **positive and significant**
(pseudo *p* < 0.01 under 999 permutations), so yield is **not** spatially
random — high-yield counties tend to neighbour other high-yield counties. The
local maps localise this: a coherent **High-High** cluster (hot spots) and a
**Low-Low** cluster (cold spots), plus a handful of **spatial outliers**.

![Placeholder LISA / Gi* cluster map](outputs/.gitkeep)
<!-- Running `make analyze` writes outputs/esda_result.gpkg and summary.json;
     render the LISA/Gi* map from the notebook and drop the PNG here. -->

```
Global Moran's I : 0.41   E[I] = -0.0102   z = 7.8   p_sim = 0.001 (999 perms)
LISA quadrants   : HH=15  LL=12  LH=2  HL=1  ns=69
Getis-Ord Gi*    : hot=14  cold=11  ns=74   (significance = 0.05)
```

*(Numbers above are illustrative placeholders; run the pipeline to regenerate
them for the configured year/state.)*

### What this analysis does **not** let you conclude

- **Not causal.** Clusters describe *where* values co-locate, not *why*. Soil,
  climate, irrigation, and management are plausible drivers but are not tested
  here.
- **Conditional on the weights matrix.** Change Queen → KNN → distance band and
  the clusters can shift. We therefore **compare** weights and report neighbour
  diagnostics rather than trusting one.
- **Conditional on stationarity.** A single global Moran's I assumes one process
  over the whole map. GWR is offered precisely because that assumption is often
  wrong.
- **Multiple comparisons.** LISA/Gi\* test every unit; pseudo *p*-values are
  descriptive flags and should be FDR/Bonferroni-corrected before strong claims.
- **MAUP.** Results depend on the areal units (counties); a different
  aggregation could change the story.

---

## How it works

```
data/download.py     # USDA NASS yield  +  TIGER counties  -> data/raw/*.gpkg
        |
src/hotspots/
  weights.py         # Queen / DistanceBand / KNN, row-standardise, island checks
  esda.py            # Moran's I (perms), LISA, Getis-Ord Gi*, significance masks
  gwr.py             # optional GWR with bandwidth selection (guarded import)
  cli.py             # `hotspots` console entry point -> outputs/summary.json
```

The numeric core is a pure-numpy reference layer with no third-party
dependency: global Moran's I (`morans_i_dense`), Geary's C (`gearys_c_dense`),
Local Moran / LISA (`local_moran_dense`, `lisa_quadrants`), and standardised
Getis-Ord Gi* (`getis_ord_g_star_dense`). It is covered by **hand-derived
known-answer tests** whose expected values are checked with exact rational
arithmetic — a monotone path graph gives *I = 1/3* and Geary *C = 3/10*; a
checkerboard gives *I = -1* and *C = 3/2*; a one-hot corner gives Gi* z-scores
of *1/√3* and *-√3*. These reference functions return point estimates only. The
pysal-backed wrappers (`global_moran`, `local_moran`, `getis_ord_gi_star`) add
the permutation inference real analyses need and import `esda` lazily, so the
core and the test suite run without the geospatial stack installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: install, NASS key, data
download, weights comparison, reading the Moran scatterplot and the LISA/Gi*
cluster maps, and a section on what these tests do not prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves the geospatial stack)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run download       # needs NASS_API_KEY (free: quickstats.nass.usda.gov/api)
pixi run analyze
pixi run test
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
export NASS_API_KEY=...        # for the download step
make download
make analyze
make test
```

### Option C — Docker

```bash
docker build -t spatial-hotspots .
docker run --rm spatial-hotspots        # runs the test suite
```

---

## Configuration

Everything analysis-defining lives in [`config/aoi.yaml`](config/aoi.yaml):
study area (state FIPS, CRS), variable (USDA NASS query), weights type and
parameters, significance level, permutations, and the optional GWR block.

---

## Data sources

- **USDA NASS QuickStats** (primary variable) — county crop yield via the public
  API. Free key required.
- **TIGER/Line counties** (areal units) — US Census county polygons.
- **Landsat C2 L2 Surface Temperature** (documented alternative) — for an
  urban-heat-island study, fetched via STAC / `earthaccess`, scaled to °C, and
  aggregated to tracts/grid by zonal statistics. The ESDA pipeline is identical;
  only the loader changes (see `data/download.py:_landsat_alternative_note`).

Raw data and outputs are git-ignored and regenerated by the download script and
pipeline.

---

## License

MIT © 2026 Joseph Mbuh
