"""Read places / facilities CSVs and (lazily) build GeoDataFrames.

The CSV readers are plain pandas: they require ``lat`` and ``lon`` columns and
return a tidy frame. :func:`to_geodataframe` lazily imports geopandas so the
numeric core and the tests stay free of any geospatial dependency -- only the
Streamlit/leafmap app calls it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LAT_LON = ("lat", "lon")


def _read_points(path: str | Path, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in LAT_LON if c not in df.columns]
    if missing:
        raise KeyError(f"{kind} CSV {path} is missing required columns: {missing}")
    return df


def load_places(path: str | Path) -> pd.DataFrame:
    """Read a populated-places CSV (requires ``lat``, ``lon``; usually ``population``)."""
    return _read_points(path, "places")


def load_facilities(path: str | Path) -> pd.DataFrame:
    """Read a health-facilities CSV (requires ``lat``, ``lon``)."""
    return _read_points(path, "facilities")


def to_geodataframe(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat"):
    """Build a point GeoDataFrame from a lat/lon frame (lazy geopandas import).

    Used only by the app; kept out of the tested core. Returns a GeoDataFrame in
    EPSG:4326 with point geometry from ``lon_col``/``lat_col``.
    """
    import geopandas as gpd

    missing = [c for c in (lon_col, lat_col) if c not in df.columns]
    if missing:
        raise KeyError(f"frame is missing geometry columns: {missing}")
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
