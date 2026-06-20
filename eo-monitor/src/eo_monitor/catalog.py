"""STAC discovery: search Earth Search for Sentinel-2 L2A items over an AOI.

Items are filtered by cloud cover, sorted by acquisition datetime and capped at
``max_items`` so a clean run stays cheap and reproducible.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pystac_client import Client

from eo_monitor.config import AOI, Config, DateRange

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pystac import Item

logger = logging.getLogger(__name__)


def _aoi_bbox(aoi: AOI) -> list[float]:
    """Return a [lon_min, lat_min, lon_max, lat_max] bbox for the AOI.

    For a vector AOI the bounds are read lazily so the dependency is optional.
    """
    if aoi.bbox is not None:
        return list(aoi.bbox)
    if aoi.vector_path is None:  # pragma: no cover - guarded by config validator
        raise ValueError("AOI has neither bbox nor vector_path.")
    import geopandas as gpd  # local import: only needed for vector AOIs

    gdf = gpd.read_file(aoi.vector_path).to_crs(4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def search_items(
    *,
    stac_url: str,
    collection: str,
    aoi: AOI,
    date_range: DateRange,
    cloud_cover_max: float,
    max_items: int,
) -> list[Item]:
    """Search a STAC API and return filtered, datetime-sorted items.

    Parameters mirror the config. The ``eo:cloud_cover`` filter is pushed to the
    server via CQL2; results are sorted ascending by ``datetime`` and truncated
    to ``max_items`` as a hard guard.
    """
    bbox = _aoi_bbox(aoi)
    client = Client.open(stac_url)
    logger.info(
        "STAC search: collection=%s bbox=%s datetime=%s cloud<=%.0f%%",
        collection,
        bbox,
        date_range.as_query(),
        cloud_cover_max,
    )
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range.as_query(),
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
        max_items=max_items,
    )
    items = list(search.items())

    # Defensive client-side re-filter & sort in case the server ignores hints.
    items = [
        it for it in items if (it.properties.get("eo:cloud_cover", 100.0) < cloud_cover_max)
    ]
    items.sort(key=lambda it: it.properties.get("datetime", ""))
    if len(items) > max_items:
        items = items[:max_items]

    logger.info("Found %d items after filtering.", len(items))
    if not items:
        raise ValueError(
            "No STAC items matched. Loosen cloud_cover_max or widen the date range / AOI."
        )
    return items


def search_for_config(config: Config, window: DateRange | None = None) -> list[Item]:
    """Convenience wrapper using a :class:`Config`.

    ``window`` overrides ``config.date_range`` (used to fetch the baseline window
    with the same filters).
    """
    return search_items(
        stac_url=config.stac.url,
        collection=config.stac.collection,
        aoi=config.aoi,
        date_range=window or config.date_range,
        cloud_cover_max=config.cloud_cover_max,
        max_items=config.max_items,
    )
