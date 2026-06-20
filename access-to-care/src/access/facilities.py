"""Load, clean, and snap health-facility points to the road network.

Pipeline role: read facility points (Healthsites.io GeoJSON), drop invalid
geometries, reproject to the working CRS, and snap each facility to its nearest
graph node so it can act as a source for shortest-path routing.

The geometry-free helper :func:`clean_facility_records` is unit-testable; the
geopandas/osmnx functions require the geospatial stack and a real graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def clean_facility_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter raw facility records to those with usable coordinates.

    Each record must have finite numeric ``lon``/``lat`` within valid WGS84
    ranges. Returns a new list of cleaned records (no mutation). Pure Python so
    the cleaning rules can be tested without geo dependencies.
    """
    cleaned: list[dict[str, Any]] = []
    for rec in records:
        lon = rec.get("lon")
        lat = rec.get("lat")
        if lon is None or lat is None:
            continue
        try:
            lon_f = float(lon)
            lat_f = float(lat)
        except (TypeError, ValueError):
            continue
        if not (-180.0 <= lon_f <= 180.0 and -90.0 <= lat_f <= 90.0):
            continue
        if lon_f == 0.0 and lat_f == 0.0:
            continue  # null-island sentinel
        out = dict(rec)
        out["lon"] = lon_f
        out["lat"] = lat_f
        cleaned.append(out)
    return cleaned


def geojson_features_to_records(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten GeoJSON point features into ``lon``/``lat`` records.

    Reads the Healthsites.io FeatureCollection shape: each feature carries a
    ``geometry`` of type ``Point`` with ``coordinates`` ``[lon, lat]`` and a
    ``properties`` dict. Non-point features and features without coordinates are
    skipped. Property keys are copied alongside ``lon``/``lat`` so downstream
    cleaning (:func:`clean_facility_records`) sees one flat record per facility.
    Pure Python; no geopandas needed.

    Parameters
    ----------
    features : list of dict
        GeoJSON feature objects.

    Returns
    -------
    list of dict
        One record per usable point feature, each with ``lon`` and ``lat`` and
        the feature's properties.
    """
    records: list[dict[str, Any]] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        rec: dict[str, Any] = dict(props) if isinstance(props, dict) else {}
        rec["lon"] = coords[0]
        rec["lat"] = coords[1]
        records.append(rec)
    return records


def load_facilities(path: str | Path, target_crs: str = "EPSG:32632"):
    """Read a facility GeoJSON into a cleaned, reprojected GeoDataFrame."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(target_crs)
    gdf = gdf.reset_index(drop=True)
    return gdf


def snap_facilities_to_nodes(facilities, graph):
    """Snap facility points to their nearest graph node.

    Returns the facilities GeoDataFrame with a new ``node_id`` column. Requires
    osmnx and a graph whose nodes carry ``x``/``y`` (projected) coordinates.
    """
    import osmnx as ox

    xs = facilities.geometry.x.to_numpy()
    ys = facilities.geometry.y.to_numpy()
    node_ids = ox.distance.nearest_nodes(graph, X=xs, Y=ys)
    out = facilities.copy()
    out["node_id"] = node_ids
    return out


def facility_source_nodes(facilities) -> list[Any]:
    """Return the unique set of node ids used as routing sources."""
    if "node_id" not in facilities.columns:
        raise KeyError("facilities must be snapped first (call snap_facilities_to_nodes)")
    return list(dict.fromkeys(facilities["node_id"].tolist()))
