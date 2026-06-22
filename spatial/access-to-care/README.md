# access-to-care

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)

> **How far is the nearest clinic for the people of this region?**

Travel-time-to-care accessibility and equity analysis. Given a road network and
health-facility locations, this pipeline computes how long it takes each
populated place to reach the nearest facility, weights those times by population,
and reports which areas have low coverage at the 30/60/120-minute thresholds.

## The result

The full pipeline needs the real WorldPop / Healthsites / OSM inputs. To show the
machinery end to end without those downloads, a self-contained demo runs the real
routing and equity code over a **small seeded synthetic road network** (a 12×12
grid, three facilities, a population per node). Its numbers are reproducible but
illustrative — not measured from Cameroon data.

**Demo output (`seed=0`, 144 nodes, 3 facilities, total population ≈ 107,696):**

| Metric | Value |
| --- | --- |
| Population within 30 min | 10.0% |
| Population within 60 min | 35.1% |
| Population within 120 min | 93.3% |
| Population unreachable | 0 |
| Gini of travel time (population-weighted) | 0.258 |

Coverage bands (people per disjoint travel-time bucket): 0–30 min ≈ 10,778;
30–60 min ≈ 26,995; 60–120 min ≈ 62,671; 120 min+ ≈ 7,252.

**Reproduce:** `pixi run demo` (or `make demo`, or
`python -m access.demo`). It writes `outputs/demand.csv` and
`outputs/summary.json`.

*A choropleth of population-weighted travel time by admin-2 unit is written to
`outputs/` when the full pipeline runs on real inputs.*

## How it works

1. **Network** (`src/access/network.py`) — load a drivable graph from an OSM
   extract, set each edge's `travel_time` from its highway type and length, and
   keep the largest connected component.
2. **Facilities** (`src/access/facilities.py`) — clean health-facility points,
   reproject, and snap each to its nearest network node.
3. **Access** (`src/access/access.py`) — multi-source Dijkstra from all facility
   nodes over `travel_time`, assigning every demand cell (H3 hexes or a
   population grid) the time to its nearest facility.
4. **Equity** (`src/access/equity.py`) — join travel times to WorldPop
   population and summarise, per GADM admin-2 unit, the population within
   30/60/120 minutes and the share left beyond each threshold, plus a national
   figure. `aggregate_admins_to_national` rolls the admin table back up to the
   national figure; `coverage_bands` reports population in disjoint travel-time
   bands plus an unreachable bucket.

The core routing and equity arithmetic are pure Python / numpy / pandas, so the
logic is fully unit-tested (`tests/`) without needing the heavy geospatial stack.

## Capabilities

Beyond the within-threshold coverage shares, the pure core (`src/access/`) also
computes:

- **Gini coefficient** (`metrics.gini_coefficient`) — population-weighted
  inequality of travel times, 0 (everyone equal) to 1 (maximally unequal).
- **2SFCA** (`metrics.two_step_floating_catchment`) — the classic two-step
  floating catchment area accessibility score, combining facility capacity and
  demand within a travel-time catchment.
- **Facility load** (`metrics.facility_load`) — the catchment demand
  (population) assigned to each facility, built on
  `access.assign_nearest_source`, a multi-source Dijkstra that returns both the
  cost and *which* facility is nearest per node.
- **Coverage bands** (`equity.coverage_bands`) — population in disjoint
  travel-time buckets plus an unreachable bucket that partition the total.

Each has a hand-derived known-answer test (`tests/test_metrics.py`).

## Data

All inputs are downloaded reproducibly into `data/raw/` (git-ignored). URLs and
study-area parameters live in [`config/sources.yaml`](config/sources.yaml):

| Dataset | Source |
| --- | --- |
| Roads + amenities | Geofabrik OSM extract (Cameroon) |
| Health facilities | Healthsites.io API |
| Population (100m) | WorldPop constrained 2020 |
| Admin-2 boundaries | GADM v4.1 |

## How to run

The seeded demo needs only numpy/pandas — no downloads, no geospatial stack:

```bash
pixi run demo            # or: make demo, or python -m access.demo
```

The full pipeline uses [pixi](https://pixi.sh) to manage the conda-forge
geospatial stack (GDAL/GEOS/PROJ resolve cleanly that way).

```bash
# 1. Install the environment. `pixi install` resolves pixi.toml and GENERATES
#    pixi.lock (the lockfile is not committed by hand).
pixi install

# 2. Download all inputs into data/raw/ (requires a network connection).
pixi run download          # or: python data/download.py

# 3. Build the network + accessibility surface.
pixi run access data/raw/cameroon-latest.osm.pbf data/raw/healthsites_cameroon.geojson

# 4. Tests / lint
pixi run test
pixi run lint
```

Pip fallback (no conda system libraries managed for you):

```bash
pip install -r requirements.txt
export HEALTHSITES_API_KEY="your-key"   # needed for the facilities download
python data/download.py
python -m access.access \
    data/raw/cameroon-latest.osm.pbf \
    data/raw/healthsites_cameroon.geojson \
    --out outputs/access.gpkg
pytest -q
```

The pure-Python routing and equity tests run with only numpy/pandas installed
(no geo stack), so `pytest -q` passes before the heavy dependencies are present.

For the full walkthrough — environment, the `HEALTHSITES_API_KEY`, what each
download fetches and where it lands, the network → facilities → access → equity
steps, expected outputs, and troubleshooting — see [`USAGE.md`](USAGE.md).

Walk through the narrative end-to-end in
[`notebooks/01_access_story.ipynb`](notebooks/01_access_story.ipynb).

## License

MIT © 2026 Joseph Mbuh — see [LICENSE](LICENSE).
