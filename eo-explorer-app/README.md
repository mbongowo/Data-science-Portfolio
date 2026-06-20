# 🛰️ EO Explorer

> **Live demo:** TODO — add the deployed URL here after the first deploy.
> Target hosts: Streamlit Community Cloud, Hugging Face Spaces, or Fly.io.

<!-- TODO (after deploy): replace docs/screenshot.png with a real screenshot of the running app. -->
![EO Explorer screenshot](docs/screenshot.png)

[![CI](https://img.shields.io/github/actions/workflow/status/JosephMbuh/eo-explorer-app/ci.yml?branch=main&label=CI)](https://github.com/JosephMbuh/eo-explorer-app/actions)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/built%20with-streamlit-FF4B4B)](https://streamlit.io)

A **shippable, interactive Earth-observation web app**. Draw an area of interest
on the map, pick a date and a spectral index, and the app pulls **live
Sentinel-2 L2A** imagery from a public STAC catalogue and renders the index on
an interactive map — with a colour-bar legend and summary stats.

This is **Project 6** of a 6-repo data-science portfolio. The point is to
prove I can *ship*, not just analyse, and that I package my own code as a
reusable dependency: the index maths is **reused from the flagship
[`eo-monitor`](https://github.com/JosephMbuh/eo-monitor) package**, not
copy-pasted.

---

## What it does

1. **Draw an AOI.** Use the rectangle/polygon draw control on the leafmap map.
2. **Pick a date** — the app searches ±10 days for the *least-cloudy* scene.
3. **Pick an index** — NDVI (vegetation), NDWI (water), or NDMI (moisture).
4. **Render.** The app queries [Earth Search](https://earth-search.aws.element84.com/v1)
   (Sentinel-2 L2A, no auth), loads only the bands it needs via `odc-stac`,
   computes the index with `eo-monitor`, and adds a coloured raster layer with a
   legend and min/mean/max statistics.

Edge cases surface a single specific message rather than failing silently:
oversized AOIs are rejected by `validate_aoi`, empty search results and
out-of-range dates each produce a distinct warning, and STAC queries/loads are
cached (`st.cache_data`) keyed by **AOI bbox + date + index**.

## How it reuses `eo-monitor`

`app/render.py` imports the index functions directly from the flagship package:

```python
from eo_monitor.indices import ndvi, ndwi, ndmi
```

The functions are **not reimplemented**. `render.require_eo_monitor()` raises a
clear, actionable error if the package isn't installed, so the dependency is
explicit and visible. This makes the portfolio projects *compose*: the analysis
library (`eo-monitor`) becomes the engine inside a deployed product
(`eo-explorer-app`).

Install the dependency one of these ways:

```bash
# Local editable (sibling repo at ../eo-monitor) -- recommended for dev:
pip install -e ../eo-monitor

# Or straight from git:
pip install "eo-monitor @ git+https://github.com/JosephMbuh/eo-monitor.git"

# Or via the optional extra declared in pyproject.toml:
pip install "eo-explorer-app[eo-monitor-git]"
```

## Run locally

### With pixi (recommended)

```bash
pixi install            # resolves deps and GENERATES pixi.lock
pixi run install-eo-monitor   # pip install -e ../eo-monitor
pixi run app            # streamlit run app/main.py
```

> The `pixi.lock` file is created by `pixi install` — it is intentionally **not**
> committed by hand.

### With pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ../eo-monitor       # the reused flagship package
streamlit run app/main.py
```

### With Docker

```bash
docker build -t eo-explorer-app .
docker run -p 8501:8501 eo-explorer-app
# open http://localhost:8501
```

## Tests

`tests/test_smoke.py` covers the **pure helpers** and the colour-mapping layer:
`aoi_bbox_from_geojson` across Polygon, MultiPolygon, Feature, and
FeatureCollection inputs; `validate_aoi` against zero area, inverted boxes,
out-of-range and non-numeric coordinates, antimeridian crossings, and the
area cap; `cache_key` determinism and collision resistance; `date_window`
boundary cases; and `normalize`/`colorize` against a matplotlib colormap. The
suite runs with only stdlib, numpy, and matplotlib, so no streamlit, leafmap,
or STAC stack is needed:

```bash
pytest -q
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, and the
test suite on Python 3.10 and 3.12.

For installation, the UI walkthrough, caching behaviour, the edge-case
messages, and per-host deployment steps, see [`USAGE.md`](USAGE.md).

## Deploy notes

Deployment is a **manual step** (pick one):

- **Streamlit Community Cloud** — point it at this repo and `app/main.py`. Uncomment
  the `eo-monitor` git line in `requirements.txt` so the index code is available,
  then put the live URL at the top of this README.
- **Hugging Face Spaces.** Create a *Streamlit* Space, push this repo, and make sure
  the `eo-monitor` git line in `requirements.txt` is uncommented.
- **Fly.io** — `fly launch` with the provided `Dockerfile`, which installs
  `eo-monitor` from git and `EXPOSE`s 8501.

## Project layout

```
eo-explorer-app/
├── app/
│   ├── __init__.py
│   ├── main.py      # Streamlit entry: leafmap map + draw control + sidebar
│   ├── stac.py      # Earth Search query + load; pure AOI/cache helpers
│   └── render.py    # compute index via eo-monitor + coloured layer + legend
├── tests/test_smoke.py
├── USAGE.md         # install, UI walkthrough, caching, deployment steps
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── pyproject.toml   # deps incl. eo-monitor (path/git)
├── pixi.toml        # conda-forge env (pixi install -> pixi.lock)
├── requirements.txt # pip fallback
├── Dockerfile · Makefile · .pre-commit-config.yaml · .gitignore · LICENSE
```

## Data & licensing

- **Imagery:** Sentinel-2 L2A via Earth Search (Element 84 / AWS Open Data),
  fetched **live** — no data is committed to the repo.
- **Code:** MIT © 2026 Joseph Mbuh.
