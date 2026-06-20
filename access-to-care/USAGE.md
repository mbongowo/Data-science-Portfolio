# Usage guide

End-to-end instructions for running the access-to-care pipeline: install the
environment, fetch the inputs, build the road network, compute nearest-facility
travel times, and produce the population-weighted equity summary for Cameroon.

The routing and equity arithmetic are pure Python/numpy/pandas and run without
any geospatial libraries (see `tests/`). The full pipeline that reads OSM,
rasters, and admin boundaries needs the geospatial stack (osmnx, geopandas,
shapely, pyproj, h3, rasterio, rasterstats), which resolves most reliably from
conda-forge.

## 1. Install

The geospatial stack depends on GDAL/GEOS/PROJ system libraries. pixi pulls
those from conda-forge and pins them in a lockfile.

```bash
pixi install        # resolves pixi.toml, generates pixi.lock
```

Pip fallback (you manage GDAL/GEOS/PROJ yourself; on Windows this often means
wheels from a maintained index or a conda base):

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Supported Python is 3.10–3.12. The pure-Python tests also run on 3.13/3.14, but
osmnx/geopandas wheels may not yet exist there, so use 3.10–3.12 for the full
pipeline.

Confirm the import surface:

```bash
python -c "import osmnx, geopandas, shapely, rasterio, h3; print('geo stack ok')"
```

## 2. Set environment variables

Healthsites.io serves health-facility points behind an API key. Register at
<https://healthsites.io>, create a key, and export it before downloading:

```bash
export HEALTHSITES_API_KEY="your-key-here"     # Windows PowerShell: $env:HEALTHSITES_API_KEY="your-key-here"
```

The key name is configurable in `config/sources.yaml` under
`sources.facilities.api_key_env`. Geofabrik (OSM), WorldPop (population), and
GADM (admin boundaries) need no key. If `HEALTHSITES_API_KEY` is unset the
download still attempts the request, but the API returns fewer records or rejects
the call.

## 3. Download the inputs

```bash
python data/download.py                  # all four sources
python data/download.py --only osm facilities
python data/download.py --force          # re-fetch even if present
```

Everything lands in `data/raw/` (git-ignored). Files already present are skipped
unless `--force` is given.

| Key | File | Approx. size | Notes |
| --- | --- | --- | --- |
| `osm` | `cameroon-latest.osm.pbf` | ~150–250 MB | Geofabrik national extract, updated daily |
| `facilities` | `healthsites_cameroon.geojson` | a few MB | Paginated; pages merge into one FeatureCollection |
| `population` | `cmr_ppp_2020_UNadj_constrained.tif` | ~100–300 MB | WorldPop 100 m constrained, UN-adjusted, 2020 |
| `admin` | `gadm41_CMR.gpkg` | ~30–60 MB | GADM v4.1, admin-2 layer `ADM_ADM_2` |

The OSM and WorldPop downloads dominate the time and disk budget; expect several
minutes on a typical connection. URLs and filenames are read from
`config/sources.yaml`, so edit that file rather than the script to change a
source.

## 4. Build the network

`src/access/network.py` reads the OSM extract, assigns each edge a
`travel_time` (seconds) from its `highway` tag and length using the speed table
in `config/sources.yaml`, and keeps the largest strongly connected component.

```bash
python -m access.network data/raw/cameroon-latest.osm.pbf
# prints: Graph: <N> nodes, <M> edges
```

Edge speeds come from `speeds_kph` in the config; unknown highway tags fall back
to `default_speed_kph`. To bias the travel-time field (for example, slower rural
speeds), edit those values.

## 5. Facilities, access, and equity

The combined entry point builds the graph, snaps facilities to nodes, builds the
demand surface, and writes travel times per demand cell:

```bash
python -m access.access \
    data/raw/cameroon-latest.osm.pbf \
    data/raw/healthsites_cameroon.geojson \
    --out outputs/access.gpkg
```

What happens, step by step:

1. **Facilities** (`src/access/facilities.py`) — read the GeoJSON, drop empty or
   non-point geometries and out-of-range coordinates, reproject to the projected
   CRS (`EPSG:32632`, UTM 32N), and snap each facility to its nearest graph node.
   Those nodes are the routing sources.
2. **Demand surface** (`src/access/access.py`) — build either H3 hexagon
   centroids (`demand.type: h3`, resolution 7 by default) or a regular point grid
   (`demand.type: grid`) clipped to the study area.
3. **Access** — multi-source Dijkstra from all facility nodes over `travel_time`,
   then assign every demand cell the travel time (minutes) to its nearest
   facility. Cells whose snapped node cannot reach any facility get `NaN`.
4. **Equity** (`src/access/equity.py`) — sample WorldPop population onto demand
   cells, spatial-join cells to admin-2 polygons, and summarise. The arithmetic
   functions are `population_within_thresholds`, `summarise_by_admin`,
   `national_summary`, `aggregate_admins_to_national`, and `coverage_bands`.

The equity functions take a tidy demand frame (`admin2`, `travel_time_min`,
`population`) and need no geospatial library, so you can run the summary in a
notebook on a frame you assemble yourself:

```python
from access.equity import summarise_by_admin, national_summary

per_admin = summarise_by_admin(demand_df, thresholds_min=[30, 60, 120])
national  = national_summary(demand_df, thresholds_min=[30, 60, 120])
```

`aggregate_admins_to_national(per_admin, [30, 60, 120])` rolls the admin table
back up and reproduces `national_summary` exactly; `coverage_bands` splits the
population into disjoint travel-time bands plus an unreachable bucket that sum to
the total.

## 6. Expected outputs

- `outputs/access.gpkg` — demand cells with `node_id` and `travel_time_min`.
- A per-admin-2 summary table: `population_total`, `pop_within_{30,60,120}min`,
  and `share_within_/share_beyond_` columns per threshold. The within and beyond
  shares sum to 1 at each threshold, and within-shares are non-decreasing as the
  threshold grows.
- A national figure with the same columns.
- A choropleth of population-weighted travel time by admin-2 unit, written to
  `outputs/`. The notebook `notebooks/01_access_story.ipynb` walks through map and
  table generation.

## Troubleshooting

**Download fails or stalls.** Re-run `python data/download.py --only <key>`;
partial files are written to a `.part` sidecar and only renamed on completion, so
an interrupted download will not leave a corrupt target. A 401/403 on the
facilities source means `HEALTHSITES_API_KEY` is unset or invalid. Geofabrik and
WorldPop links occasionally move; check the URL in `config/sources.yaml` against
the provider's site if you get a 404.

**OSM parsing errors.** osmnx reads `.osm`/`.osm.xml` via `graph_from_xml`. A raw
`.pbf` may need conversion (osmium, `osmconvert`) to XML, or a newer osmnx that
reads PBF directly. If the graph comes back tiny, the extract may be clipped to a
bounding box; check `study_area.bbox` in the config (set it to `null` for the full
national extent).

**CRS mismatches.** Distances and snapping run in the projected CRS
(`EPSG:32632`); WorldPop sampling and H3 cell generation run in the geographic
CRS (`EPSG:4326`). Facility GeoJSON without a declared CRS is assumed WGS84. If
travel times look implausibly large or small, confirm `projected_crs` suits your
study area (UTM 32N covers most of Cameroon; a different country needs its own
zone).

**Missing or misaligned population tiles.** WorldPop ships one raster per country
per year. If population samples come back as zero everywhere, check that the
raster covers the demand extent and that the demand points were reprojected to the
raster's CRS before sampling. A demand cell outside the raster footprint samples
as nodata and is treated as zero population.

**Unreachable demand cells.** Cells whose nearest graph node sits on a
disconnected fragment get `travel_time_min = NaN`. The equity functions count NaN
(and inf) as beyond every threshold, so a spike in "beyond 120 min" or
`pop_unreachable` usually points to network fragmentation upstream rather than a
real access gap. Keeping the largest strongly connected component
(`largest_component`) reduces this.
