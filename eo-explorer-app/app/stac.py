"""STAC querying and scene loading for the EO Explorer app.

This module talks to the Earth Search STAC API (Sentinel-2 L2A, no auth
required), finds the least-cloudy scene intersecting a drawn AOI near a chosen
date, and loads the required bands into an :class:`xarray.Dataset`.

Design notes
------------
The *pure* helpers near the top of this module have **no third-party
dependencies** and are unit-tested in ``tests/test_smoke.py``:

* :func:`aoi_bbox_from_geojson` - turn a drawn GeoJSON geometry into a bbox.
* :func:`validate_aoi` - reject oversized or malformed AOIs with a specific message.
* :func:`cache_key` - build a deterministic cache key.

The heavier functions (:func:`find_scene`, :func:`load_scene`) import
``pystac_client``, ``odc.stac`` and friends lazily so that the pure helpers (and
the smoke tests) keep working with only the standard library + numpy installed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Earth Search v1 STAC endpoint (Element 84, AWS Open Data, no auth).
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

#: Collection name for Sentinel-2 Level-2A surface reflectance.
SENTINEL2_COLLECTION = "sentinel-2-l2a"

#: Default maximum AOI area we are willing to load (square kilometres).
DEFAULT_MAX_AREA_KM2 = 2_500.0

#: Mapping from index name -> Sentinel-2 asset (band) keys required to compute it.
#: NDVI  = (NIR - Red)   / (NIR + Red)
#: NDWI  = (Green - NIR) / (Green + NIR)    (McFeeters water index)
#: NDMI  = (NIR - SWIR1) / (NIR + SWIR1)
INDEX_BANDS: dict[str, tuple[str, ...]] = {
    "NDVI": ("red", "nir"),
    "NDWI": ("green", "nir"),
    "NDMI": ("nir", "swir16"),
}

#: How many days either side of the chosen date to search for a scene.
DEFAULT_DATE_WINDOW_DAYS = 10


# --------------------------------------------------------------------------- #
# Pure helpers (no third-party deps -- unit tested)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AOIValidation:
    """Result of validating an AOI bounding box."""

    ok: bool
    area_km2: float
    message: str


def _coords_iter(geometry: dict[str, Any]) -> Iterable[Sequence[float]]:
    """Yield every ``(lon, lat)`` coordinate pair in a GeoJSON geometry.

    Handles Point, LineString, Polygon, Multi* and GeometryCollection by walking
    the nested coordinate arrays until it reaches pairs of numbers.
    """

    def walk(node: Any) -> Iterable[Sequence[float]]:
        if (
            isinstance(node, (list, tuple))
            and len(node) >= 2
            and all(isinstance(v, (int, float)) for v in node[:2])
        ):
            yield (float(node[0]), float(node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                yield from walk(child)

    if geometry.get("type") == "GeometryCollection":
        for geom in geometry.get("geometries", []):
            yield from walk(geom.get("coordinates", []))
    else:
        yield from walk(geometry.get("coordinates", []))


def aoi_bbox_from_geojson(geojson: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return ``(min_lon, min_lat, max_lon, max_lat)`` for a GeoJSON object.

    The function walks every coordinate it can reach and takes the extremes, so
    it does not care which geometry type produced them. A leafmap draw control
    may emit a bare Polygon, a Feature wrapping one, or a FeatureCollection of
    several shapes; a MultiPolygon and a GeometryCollection work the same way.

    Parameters
    ----------
    geojson : dict
        A GeoJSON Feature, FeatureCollection, or geometry mapping.

    Returns
    -------
    tuple of float
        ``(min_lon, min_lat, max_lon, max_lat)`` covering every coordinate
        found. For a FeatureCollection this is the union extent of all features.

    Raises
    ------
    ValueError
        If ``geojson`` is not a dict, has no ``type``, or contains no usable
        coordinate pairs.
    """
    if not isinstance(geojson, dict):
        raise ValueError("geojson must be a dict")

    geometries: list[dict[str, Any]] = []
    gtype = geojson.get("type")
    if gtype == "FeatureCollection":
        geometries = [
            feat.get("geometry", {}) for feat in geojson.get("features", [])
        ]
    elif gtype == "Feature":
        geometries = [geojson.get("geometry", {})]
    elif gtype is not None:
        geometries = [geojson]
    else:
        raise ValueError("geojson is missing a 'type' field")

    lons: list[float] = []
    lats: list[float] = []
    for geom in geometries:
        if not geom:
            continue
        for lon, lat in _coords_iter(geom):
            lons.append(lon)
            lats.append(lat)

    if not lons or not lats:
        raise ValueError("No coordinates found in the supplied GeoJSON")

    return (min(lons), min(lats), max(lons), max(lats))


def bbox_area_km2(bbox: Sequence[float]) -> float:
    """Approximate the area of a lon/lat bbox in square kilometres.

    Uses an equirectangular approximation that scales longitude spacing by the
    cosine of the mean latitude. The result gates AOI size, so a few percent of
    error from the approximation does not matter.

    Parameters
    ----------
    bbox : sequence of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.

    Returns
    -------
    float
        Area in square kilometres. Always non-negative.

    Raises
    ------
    ValueError
        If the ordering is wrong (``max_lon < min_lon`` or ``max_lat <
        min_lat``). A west-to-east bbox that crosses the antimeridian is one
        legitimate case where ``max_lon < min_lon``; detect that earlier with
        :func:`crosses_antimeridian` rather than calling this function.
    """
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if max_lon < min_lon or max_lat < min_lat:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")

    mean_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(mean_lat_rad)

    height_km = (max_lat - min_lat) * km_per_deg_lat
    width_km = (max_lon - min_lon) * km_per_deg_lon
    return abs(height_km * width_km)


def crosses_antimeridian(bbox: Sequence[float]) -> bool:
    """Report whether a bbox appears to straddle the +/-180 deg antimeridian.

    A draw control that wraps across the date line emits a bbox whose western
    edge sits at a larger longitude than its eastern edge (for example
    ``min_lon=170`` and ``max_lon=-170``). The STAC search and the area estimate
    both assume ``min_lon <= max_lon``, so such a bbox is rejected upstream.

    Parameters
    ----------
    bbox : sequence of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.

    Returns
    -------
    bool
        ``True`` when the longitudes are out of order, which signals an
        antimeridian crossing. ``False`` for any well-formed bbox, including one
        of length other than four (those fail other checks first).
    """
    coords = tuple(float(v) for v in bbox)
    if len(coords) != 4:
        return False
    min_lon, _min_lat, max_lon, _max_lat = coords
    return max_lon < min_lon


def validate_aoi(
    bbox: Sequence[float], max_area_km2: float = DEFAULT_MAX_AREA_KM2
) -> AOIValidation:
    """Validate an AOI bounding box and return a UI-ready verdict.

    The checks run in this order: shape (four numbers), longitude/latitude
    ranges, antimeridian crossing, ordering, zero or near-zero extent, then the
    maximum-area limit. The first failure wins, so the returned message names a
    single concrete problem.

    Parameters
    ----------
    bbox : sequence of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
    max_area_km2 : float, optional
        Largest area the caller is willing to load. Defaults to
        :data:`DEFAULT_MAX_AREA_KM2`.

    Returns
    -------
    AOIValidation
        ``ok`` is ``True`` only when every check passes. ``area_km2`` is the
        estimated area (``0.0`` when the area could not be computed), and
        ``message`` is a sentence suitable for display.
    """
    if bbox is None:
        return AOIValidation(False, 0.0, "Please draw a rectangular area on the map.")

    coords = tuple(bbox)
    if len(coords) != 4:
        return AOIValidation(False, 0.0, "Please draw a valid rectangular area on the map.")

    try:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in coords)
    except (TypeError, ValueError):
        return AOIValidation(False, 0.0, "The drawn area has non-numeric coordinates.")

    if any(math.isnan(v) or math.isinf(v) for v in (min_lon, min_lat, max_lon, max_lat)):
        return AOIValidation(False, 0.0, "The drawn area has missing or infinite coordinates.")

    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        return AOIValidation(False, 0.0, "Longitudes must be between -180 and 180 degrees.")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        return AOIValidation(False, 0.0, "Latitudes must be between -90 and 90 degrees.")

    if crosses_antimeridian(coords):
        return AOIValidation(
            False,
            0.0,
            "The area crosses the +/-180 deg date line, which this app cannot load. "
            "Draw an area that stays on one side of the antimeridian.",
        )

    if max_lat < min_lat:
        return AOIValidation(
            False, 0.0, "The drawn area is inverted (north edge below south edge)."
        )

    try:
        area = bbox_area_km2(coords)
    except ValueError as exc:
        return AOIValidation(False, 0.0, f"That area does not look right: {exc}")

    if area <= 0.0:
        return AOIValidation(
            False, area, "The drawn area has no extent - try drawing a box, not a point or line."
        )
    if area > max_area_km2:
        return AOIValidation(
            False,
            area,
            (
                f"Your area is about {area:,.0f} km^2, which is larger than the "
                f"{max_area_km2:,.0f} km^2 limit. Please draw a smaller area so the "
                "imagery loads quickly."
            ),
        )
    return AOIValidation(True, area, f"Area looks good (~{area:,.0f} km^2).")


def cache_key(bbox: Sequence[float], date: str, index: str) -> str:
    """Build a deterministic cache key for an (AOI, date, index) request.

    The key is a SHA-256 digest of a canonical JSON payload, so it is stable
    across processes (unlike the built-in ``hash()``, which is salted per run).
    The bbox is rounded to six decimal places, about 0.1 m at the equator, so
    two AOIs that differ only below that resolution share a cache entry. The
    index name is upper-cased so ``"ndvi"`` and ``"NDVI"`` collide on purpose.

    Parameters
    ----------
    bbox : sequence of float
        ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
    date : str
        The target date, normally ISO ``YYYY-MM-DD``. Stored verbatim.
    index : str
        Spectral index name; case is folded to upper.

    Returns
    -------
    str
        A key of the form ``"eo-explorer:<16 hex chars>"``. Sixteen hex
        characters give 64 bits, enough that an accidental collision between
        distinct requests is not a practical concern for an interactive app.
    """
    rounded = tuple(round(float(v), 6) for v in bbox)
    payload = {
        "bbox": rounded,
        "date": str(date),
        "index": str(index).upper(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"eo-explorer:{digest[:16]}"


def date_window(date: str, days: int = DEFAULT_DATE_WINDOW_DAYS) -> str:
    """Return a closed date interval centred on ``date`` for a STAC search.

    Parameters
    ----------
    date : str
        Centre date as ISO ``YYYY-MM-DD``.
    days : int, optional
        Half-width of the window in days. Must be zero or positive. With
        ``days=0`` the interval collapses to the single centre date repeated,
        which pystac-client reads as that one day.

    Returns
    -------
    str
        ``"<start>/<end>"`` with both endpoints in ISO date form, suitable for
        the ``datetime`` argument of a pystac-client search, e.g.
        ``"2024-06-05/2024-06-25"``.

    Raises
    ------
    ValueError
        If ``date`` is not a valid ISO date, or if ``days`` is negative.

    Notes
    -----
    Crossing a month or year boundary is handled by :class:`datetime.date`
    arithmetic, so a centre near the start of a month produces a start date in
    the previous month without special casing.
    """
    from datetime import date as _date
    from datetime import timedelta

    if days < 0:
        raise ValueError("days must be zero or positive")

    centre = _date.fromisoformat(str(date))
    start = centre - timedelta(days=days)
    end = centre + timedelta(days=days)
    return f"{start.isoformat()}/{end.isoformat()}"


# --------------------------------------------------------------------------- #
# Heavy functions (lazy third-party imports)
# --------------------------------------------------------------------------- #


def find_scene(
    bbox: Sequence[float],
    date: str,
    *,
    window_days: int = DEFAULT_DATE_WINDOW_DAYS,
    max_cloud: float = 80.0,
):
    """Find the least-cloudy Sentinel-2 scene intersecting ``bbox`` near ``date``.

    Returns the best :class:`pystac.Item` (lowest ``eo:cloud_cover``) or ``None``
    if no scene matches. Network access is required.

    Parameters
    ----------
    bbox:
        ``(min_lon, min_lat, max_lon, max_lat)``.
    date:
        ISO ``YYYY-MM-DD`` string. The search spans ``+/- window_days``.
    window_days:
        Half-width of the temporal search window.
    max_cloud:
        Discard scenes with cloud cover above this percentage.
    """
    from pystac_client import Client  # lazy import

    client = Client.open(EARTH_SEARCH_URL)
    search = client.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=tuple(float(v) for v in bbox),
        datetime=date_window(date, window_days),
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=50,
    )

    items = list(search.items())
    if not items:
        return None

    # Sort locally in case the server ignores ``sortby``.
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    return items[0]


def load_scene(item, bbox: Sequence[float], index: str):
    """Load the bands needed for ``index`` from ``item``, clipped to ``bbox``.

    Returns an :class:`xarray.Dataset` with one variable per required band,
    reprojected/clipped to the AOI. Network access is required.
    """
    import odc.stac  # noqa: F401  (registers the .stac accessor / load fn)
    from odc.stac import load as odc_load

    bands = INDEX_BANDS[index.upper()]
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)

    dataset = odc_load(
        [item],
        bands=bands,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        chunks={},  # dask-backed; lazy until needed
        resolution=10,
        crs="EPSG:3857",
    )
    # Collapse the (length-1) time dimension if present.
    if "time" in dataset.dims:
        dataset = dataset.isel(time=0)
    return dataset


def scene_metadata(item) -> dict[str, Any]:
    """Extract a small, display-friendly metadata dict from a STAC item."""
    props = getattr(item, "properties", {}) or {}
    return {
        "id": getattr(item, "id", "unknown"),
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "platform": props.get("platform") or props.get("constellation"),
    }
