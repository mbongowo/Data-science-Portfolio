"""Config-driven data download / loader for spatial-hotspots.

Primary path (recommended, most reproducible): join **USDA NASS QuickStats**
county-level crop yield to **TIGER/Line** county polygons. Both are public,
API-fetchable, and stable.

Alternative path (documented, not the default): **Landsat Collection 2 Level-2
Surface Temperature** for an urban-heat-island study, fetched via a STAC API or
``earthaccess``. See ``_landsat_alternative_note`` and the README.

Usage
-----
    python data/download.py --config config/aoi.yaml --out data/raw

Environment
-----------
* ``NASS_API_KEY`` — free key from https://quickstats.nass.usda.gov/api
  (required only for the live USDA fetch).

The functions degrade gracefully: if the required third-party libraries or the
network are unavailable, they raise a clear, actionable error rather than
failing obscurely. Nothing here runs at import time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
TIGER_COUNTY_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML AOI configuration."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# USDA NASS QuickStats
# --------------------------------------------------------------------------- #
def fetch_nass_yield(cfg: dict[str, Any], api_key: str | None = None) -> Any:
    """Fetch county crop-yield records from the USDA NASS QuickStats API.

    Returns a pandas DataFrame with one row per county and a numeric value
    column named ``cfg['variable']['value_column']``.
    """
    import pandas as pd
    import requests

    api_key = api_key or os.environ.get("NASS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NASS_API_KEY is not set. Get a free key at "
            "https://quickstats.nass.usda.gov/api and export it."
        )

    nass = cfg["variable"]["nass"]
    params = {
        "key": api_key,
        "commodity_desc": nass["commodity_desc"],
        "statisticcat_desc": nass["statisticcat_desc"],
        "unit_desc": nass["unit_desc"],
        "agg_level_desc": nass["agg_level_desc"],
        "year": str(nass["year"]),
        "state_alpha": nass["state_alpha"],
        "format": "JSON",
    }
    resp = requests.get(NASS_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()["data"]
    df = pd.DataFrame(rows)

    # NASS uses 'state_fips_code' + 'county_code'; build a 5-digit GEOID.
    df["GEOID"] = (
        df["state_fips_code"].astype(str).str.zfill(2)
        + df["county_code"].astype(str).str.zfill(3)
    )
    value_col = cfg["variable"]["value_column"]
    # 'Value' is a string with thousands separators; coerce to float.
    df[value_col] = (
        df["Value"].astype(str).str.replace(",", "", regex=False)
    )
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    # Drop "other counties" / district aggregates that have no real GEOID.
    df = df[df["county_code"].astype(str).str.zfill(3) != "998"]
    return df[["GEOID", value_col]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# TIGER/Line county polygons
# --------------------------------------------------------------------------- #
def fetch_tiger_counties(cfg: dict[str, Any], out_dir: Path) -> Any:
    """Download and read TIGER county polygons, subset to the AOI state.

    Returns a GeoDataFrame reprojected to ``cfg['aoi']['crs']``.
    """
    import geopandas as gpd

    state_fips = str(cfg["aoi"]["state_fips"]).zfill(2)
    gdf = gpd.read_file(TIGER_COUNTY_URL)
    gdf = gdf[gdf["STATEFP"] == state_fips].copy()
    gdf = gdf.to_crs(cfg["aoi"]["crs"])
    return gdf[["GEOID", "NAME", "geometry"]].reset_index(drop=True)


def build_dataset(config_path: str | Path, out_dir: str | Path) -> Path:
    """Fetch, join, and persist the analysis-ready GeoPackage.

    The output is ``<out_dir>/<aoi name>.gpkg`` with a single layer containing
    county polygons and the value column ready for ESDA.
    """
    cfg = load_config(config_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = cfg["variable"]["source"]
    if source != "usda_nass":
        raise NotImplementedError(
            f"Source '{source}' is documented but not wired as the primary "
            "path. See _landsat_alternative_note() and the README."
        )

    counties = fetch_tiger_counties(cfg, out_path)
    yields = fetch_nass_yield(cfg)
    merged = counties.merge(yields, on="GEOID", how="inner")

    if merged.empty:
        raise RuntimeError(
            "Join produced 0 rows. Check the year/state in config/aoi.yaml and "
            "that the NASS query returned county-level records."
        )

    target = out_path / f"{cfg['aoi']['name']}.gpkg"
    merged.to_file(target, driver="GPKG")
    return target


def _landsat_alternative_note() -> str:
    """Document the urban-heat-island (UHI) alternative data path."""
    return (
        "UHI alternative (not the default):\n"
        "1. Search Landsat Collection 2 Level-2 scenes over the study city via "
        "a STAC API (e.g. https://landsatlook.usgs.gov/stac-server) or NASA "
        "earthaccess, filtering by cloud cover and season.\n"
        "2. Read the ST_B10 surface-temperature band, apply the C2 L2 scale "
        "(0.00341802) and offset (149.0), and convert Kelvin to Celsius.\n"
        "3. Aggregate the temperature raster to areal units (census tracts via "
        "TIGER, or a regular grid) with zonal statistics.\n"
        "4. Feed the per-unit mean LST into the same ESDA pipeline. The "
        "weights/Moran/LISA/Gi* code is identical; only the loader changes."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download spatial-hotspots data.")
    parser.add_argument("--config", default="config/aoi.yaml")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args(argv)

    path = build_dataset(args.config, args.out)
    print(f"Wrote analysis-ready dataset to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
