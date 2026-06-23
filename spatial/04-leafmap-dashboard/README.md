# 🏥 Clinic-access dashboard — Cameroon

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/built%20with-streamlit-FF4B4B)](https://streamlit.io)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=mbongowo/Data-science-Portfolio&branch=main&mainModule=spatial/04-leafmap-dashboard/app/streamlit_app.py)

**▶ Live demo:** <https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/>
> Streamlit Community Cloud (free tier — it sleeps after inactivity, so the first load may take a few seconds to wake).

![Clinic-access dashboard screenshot](docs/screenshot.png)

An interactive **leafmap + Streamlit** dashboard that answers one local question:
**which populated places in Cameroon are farthest from a health facility?** Pick
an "underserved" distance threshold, see the places that fall beyond it on the
map, and read off the most underserved settlements. **Drill down by region,
division, or sub-division** to focus on one area, with administrative boundaries
drawn for context (sub-division is Cameroon's lowest official level — there is no
township boundary).

## Result first

The reproducible `demo` command synthesizes ~200 populated places and ~20
facilities inside a Cameroon bounding box (seeded), then runs the real pipeline.
(The **deployed app** instead loads **all ~2,070 real health facilities** and
~1,290 populated places for Cameroon from OpenStreetMap, plus region and division
boundaries — see Results.)

```text
clinic-access demo (seeded synthetic Cameroon points, straight-line distance)
  places=200  facilities=20
  mean nearest   = 40.6 km
  median nearest = 22.9 km
  share within 5 km  = 3.3%
  share within 10 km = 13.4%
  share beyond 25 km = 41.1%
  farthest place = 228.6 km
```

Reproduce it exactly:

```bash
python -m clinicaccess.cli demo
# writes outputs/places_access.csv and outputs/summary.json
```

So on this seeded surface, **41% of the population sits beyond 25 km** of the
nearest facility, and the single most underserved place is **229 km** away — the
kind of long tail the dashboard is built to surface.

## The question it answers

Health planners need a quick, honest first pass at coverage gaps: given where
people live and where clinics are, *which settlements are worst served, and how
many people live too far from care?* This dashboard answers that
interactively — move the threshold slider, watch the map and the metrics update,
and export the ranked list of underserved places.

## Inspired by `opengeos/leafmap`

This project is built on and credits **[`opengeos/leafmap`](https://github.com/opengeos/leafmap)**,
Qiusheng Wu's low-code library for interactive geospatial maps in Python. leafmap
does the heavy lifting of rendering the markers, basemap, and legend; this repo
makes it *my own* by wiring it to a specific local question, a reusable numeric
core, and a one-click Streamlit deploy.

## Method

The core is deliberately simple and fast:

1. **Nearest facility** — for every place, the great-circle (**haversine**)
   distance to the closest facility, computed as a vectorised brute-force
   distance matrix (`clinicaccess.distance`, `clinicaccess.access`).
2. **Coverage** — population within each of the 5 / 10 / 25 km thresholds and the
   share left beyond the largest (NaN-safe, mirroring the equity semantics of the
   sibling `access-to-care`).
3. **Farthest ranking** — the *n* places with the largest nearest-facility
   distance: the underserved tail.
4. **Distance bins** — a graduated band per place (`0-5`, `5-10`, `10-25`,
   `25+ km`) that drives the choropleth-style colours on the map.

### Straight-line, not road travel time

These distances are **straight-line**, not road travel time. That is the point:
haversine is cheap enough to recompute on every slider move, which makes this a
fast **screening** tool to flag candidate gaps. It will understate real access
where rivers, terrain, or missing roads force long detours. For rigorous
**road-network travel time** (multi-source Dijkstra over an OSM graph,
population-weighted by admin unit), see the sibling project
[`access-to-care`](../access-to-care). The two complement each other: screen
here, then route there.

## Pipeline

```mermaid
flowchart LR
    A[places CSV<br/>lat / lon / population] --> C[nearest-facility<br/>haversine distance]
    B[facilities CSV<br/>lat / lon] --> C
    C --> D[coverage within<br/>5 / 10 / 25 km]
    C --> E[farthest-N<br/>underserved places]
    C --> F[distance bins<br/>graduated colours]
    D --> G[leafmap + Streamlit<br/>dashboard]
    E --> G
    F --> G
```

## Run it locally

```bash
pip install -r requirements.txt        # core + app stack
python -m clinicaccess.cli demo        # reproduce the numbers above
streamlit run app/streamlit_app.py     # launch the dashboard
```

The dashboard loads the bundled sample data by default, so it runs out of the
box.

## Deploy

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=mbongowo/Data-science-Portfolio&branch=main&mainModule=spatial/04-leafmap-dashboard/app/streamlit_app.py)

The badge opens the Streamlit Community Cloud dialog pre-filled with this repo,
the `main` branch, and the main file path
`spatial/04-leafmap-dashboard/app/streamlit_app.py`. Open **Advanced settings**
and set **Python version 3.12** before clicking Deploy. Cloud installs from
`app/requirements.txt` (next to the entrypoint).

> **Live demo:** <https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/>

## Use your own area / data

The default is **real Cameroon data from OpenStreetMap** (≈2,070 health
facilities, ≈1,290 populated places) with region/division boundaries, but the app
is fully data-driven:

- **Upload your own CSVs** from the sidebar — a *places* CSV with `lat`, `lon`,
  `population` (and ideally a `name`), and a *facilities* CSV with `lat`, `lon`
  (and `name`). Lat/lon are in decimal degrees (WGS84).
- **Edit `config/config.yaml`** to change the default bbox, thresholds, the
  farthest-N count, or the colour bins.
- **Run the pipeline headless** on your own files:
  `python -m clinicaccess.cli report places.csv facilities.csv`.

## Results

- Demo (seeded synthetic, 200 places / 20 facilities): median nearest **22.9 km**,
  **41%** of population beyond 25 km, farthest place **229 km** (see the block
  above; full breakdown in `outputs/summary.json`).
- **Live app (all real OpenStreetMap facilities — ≈2,070 facilities / ≈1,290
  places):** mean nearest facility **21.1 km**, median **15.9 km**; ≈**72%** of
  population is within 5 km of a facility, while the farthest mapped town,
  **Bétaré-Oya**, is **~115 km** from any mapped facility — subject to OSM
  coverage gaps (see limitations).
- **Deployed app:** <https://mbongowo-dat-spatial04-leafmap-dashboardappstreamlit-app-mclndk.streamlit.app/>

## Data sources

- **Health facilities & populated places:** [OpenStreetMap](https://www.openstreetmap.org)
  via the Overpass API — © OpenStreetMap contributors, licensed under the
  [ODbL](https://opendatacommons.org/licenses/odbl/). Curated Cameroon exports of
  the same data are on [HDX (HOTOSM)](https://data.humdata.org/dataset/hotosm_cmr_health_facilities)
  and [Healthsites.io](https://healthsites.io/map?country=Cameroon).
- **Administrative boundaries** (10 regions, 60 divisions): [geoBoundaries](https://www.geoboundaries.org)
  (Cameroon ADM1 / ADM2, open data), simplified for display.
- **Basemap tiles:** OpenStreetMap / CARTO.

## Limitations

- **Straight-line distance, not road access.** Haversine ignores roads, rivers,
  and terrain, so it understates real travel burden. Use `access-to-care` for
  road-network travel time.
- **Data provenance.** The bundled CSVs are **real OpenStreetMap** data
  (© OpenStreetMap contributors, ODbL): Cameroon health facilities and populated
  places, sampled to keep the map responsive. OSM coverage is uneven — some real
  clinics are missing and some coordinates are imprecise. The separate `demo`
  command uses small *synthetic* points purely for a reproducible, dependency-free
  check.
- **Facility-list completeness.** Results are only as good as the facility
  inputs; missing clinics make places look more underserved than they are, and
  closed ones the reverse.
- **Population allocation.** Each place carries a single population figure at a
  point; sub-place distribution and seasonal movement are not modelled.
