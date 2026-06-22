# spatial-hotspots

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
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

**Question.** When a field has spatial structure planted in it, do these
statistics actually detect it? The headline numbers below come from the
**runnable demo** (`pixi run demo`), not a hand-picked example: a 12x12 rook
grid with a High-High block planted top-left, a Low-Low block bottom-right, and
unit-scale Gaussian noise everywhere (`seed=0`).

**Answer.** Global Moran's I is **strongly positive** and Geary's C is **well
below 1**, both signalling positive spatial autocorrelation against a null
expectation near zero. The local statistics localise it: LISA flags the two
planted cluster cores, and Getis-Ord Gi* finds one hot and one cold pocket.

```
n = 144 cells (12x12 rook grid, seed=0)
Global Moran's I : 0.7242   E[I] = -0.0070     (positive autocorrelation)
Geary's C        : 0.2392                       (< 1: clustering)
LISA quadrants   : HH=48  LL=37  LH=33  HL=26  ns=0
Getis-Ord Gi*    : hot=16  cold=16             (|z| > 1.96)
```

These are the **real** outputs of the pure-numpy core on a **small seeded
synthetic field on a grid** — honest about being synthetic, but reproducible to
the digit. The county-level USDA pipeline below uses the same statistics on real
data (which requires the geospatial stack and a NASS key).

**Reproduce:**

```bash
pixi run demo          # writes outputs/summary.json + outputs/lisa_labels.csv
```

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

### Capabilities (pure numpy, no third-party deps)

- **Global autocorrelation** — Moran's I (`morans_i_dense`) with its null
  expectation (`expected_morans_i`), and Geary's C (`gearys_c_dense`).
- **Local statistics** — Local Moran / LISA (`local_moran_dense`,
  `lisa_quadrants`) and standardised Getis-Ord Gi* (`getis_ord_g_star_dense`).
- **Join counts** — BB / WW / BW counts for a binary field (`join_counts_dense`).
- **Bivariate Moran's I** — one variable against the spatial lag of another
  (`bivariate_moran_dense`).
- **Moran scatter slope** — the OLS slope of lag on value, a cross-check that
  equals Moran's I on a row-standardised W (`moran_scatter_slope`).
- **FDR correction** — Benjamini-Hochberg for the many per-location tests
  (`benjamini_hochberg`).
- **Grid helpers** — rook-contiguity builder and row-standardisation
  (`rook_weights`, `row_standardize`) used by the demo.
- **Runnable demo** — `run_demo(seed, out_dir)` / `hotspots demo` /
  `pixi run demo`, a no-data reproducible run on a seeded synthetic grid.

Every reference function has a **hand-derived known-answer test** (checkerboard
join counts are all BW; bivariate Moran on a mirrored pair is `-1/3`; the
scatter slope equals Moran's I; a worked Benjamini-Hochberg p-vector).

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: install, NASS key, data
download, weights comparison, reading the Moran scatterplot and the LISA/Gi*
cluster maps, and a section on what these tests do not prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves the geospatial stack)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run demo           # dependency-free synthetic run, regenerates the numbers above
pixi run download       # needs NASS_API_KEY (free: quickstats.nass.usda.gov/api)
pixi run analyze
pixi run test
```

The `demo` task needs nothing but numpy and writes `outputs/summary.json`. The
`download`/`analyze` tasks need the full geospatial stack and a NASS key.

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
