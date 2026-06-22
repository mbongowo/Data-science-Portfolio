"""Default real-data path: Sentinel-2 L2A from the Earth Search STAC catalogue.

This is the **auth-free** way to feed the pure-numpy core with real imagery. It
queries the Element-84 **Earth Search** STAC API for Sentinel-2 L2A items over an
AOI and a date range, loads the needed bands with ``odc.stac`` (or
``rioxarray``), masks clouds with the scene-classification layer (SCL), computes
the requested index, and reduces the period to a single cloud-robust temporal
composite. No Earth Engine account, no Google sign-in, no API key.

The heavy geospatial dependencies (``pystac-client``, ``odc-stac`` / ``odc-geo``,
``rioxarray``, ``xarray``, ``rasterio``) are imported lazily *inside* the
functions, so ``import eets.stac`` never fails on a machine that only has numpy,
and the test suite never imports this module.

Earth Search Sentinel-2 L2A band asset keys: ``red`` (B04), ``green`` (B03),
``blue`` (B02), ``nir`` (B08), ``nir08`` (B8A), ``swir16`` (B11),
``swir22`` (B12), and ``scl`` (scene classification). Index -> bands:

* ``ndvi`` -> ``nir`` + ``red``
* ``ndwi`` -> ``green`` + ``nir``
* ``nbr``  -> ``nir`` + ``swir16``
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eets.indices import nbr, ndvi, ndwi
from eets.timeseries import mask_invalid, temporal_composite

# Element-84 Earth Search v1 — open, no authentication.
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

# SCL classes treated as invalid (cloud, shadow, snow, saturated, nodata).
# 0 nodata, 1 saturated/defective, 3 cloud shadow, 8 cloud medium-prob,
# 9 cloud high-prob, 10 thin cirrus, 11 snow/ice.
DEFAULT_INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)

# Asset keys needed per index.
_INDEX_BANDS: dict[str, tuple[str, str]] = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "nbr": ("nir", "swir16"),
}
_INDEX_FUNCS = {"ndvi": ndvi, "ndwi": ndwi, "nbr": nbr}


def _import_eo_stack():
    """Lazily import the heavy EO stack with a friendly error message."""
    try:
        import odc.stac  # noqa: F401
        import pystac_client
        import xarray as xr  # noqa: F401

        return odc.stac, pystac_client, xr
    except ImportError as exc:  # pragma: no cover - needs full geo stack
        raise ImportError(
            "The STAC path needs the geospatial stack (pystac-client, odc-stac, "
            "rioxarray, xarray, rasterio). Install the 'full' extra: "
            "`pip install -e .[full]` or `conda env create -f environment.yml`."
        ) from exc


def load_s2_period(
    bbox: list[float],
    start: str,
    end: str,
    max_cloud: float = 40.0,
    index: str = "ndvi",
    resolution: float = 10.0,
    invalid_scl: tuple[int, ...] = DEFAULT_INVALID_SCL,
) -> np.ndarray:
    """Load a cloud-masked temporal composite of one index for a period.

    Searches Earth Search for Sentinel-2 L2A scenes over ``bbox`` between
    ``start`` and ``end`` with scene cloud cover below ``max_cloud``, loads the
    bands the requested ``index`` needs plus the SCL, masks invalid pixels to
    ``NaN``, computes the index per scene, and returns the per-pixel temporal
    **median** composite (cloud-robust) as a 2-D numpy array.

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    start, end:
        ISO dates, e.g. ``"2018-01-01"`` / ``"2019-12-31"``.
    max_cloud:
        Maximum scene cloud cover percentage to admit.
    index:
        ``"ndvi"``, ``"ndwi"``, or ``"nbr"``.
    resolution:
        Output pixel size in metres (10 for S2 visible/NIR).
    invalid_scl:
        SCL classes to mask out.

    Returns
    -------
    numpy.ndarray
        2-D float array: the temporal-median index composite for the period.
    """
    if index not in _INDEX_BANDS:
        raise ValueError(
            f"unknown index {index!r}; expected one of {list(_INDEX_BANDS)}"
        )
    odc_stac, pystac_client, _ = _import_eo_stack()

    band_a, band_b = _INDEX_BANDS[index]
    catalog = pystac_client.Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("no Sentinel-2 items found for AOI / date range")

    ds = odc_stac.load(
        items,
        bands=[band_a, band_b, "scl"],
        bbox=bbox,
        resolution=resolution,
        groupby="solar_day",
        chunks={},
    )

    # Mask each band with the SCL, per time step, then stack the per-scene index.
    a = ds[band_a].astype("float32").to_numpy()
    b = ds[band_b].astype("float32").to_numpy()
    scl = ds["scl"].to_numpy()

    index_func = _INDEX_FUNCS[index]
    per_scene = []
    for t in range(a.shape[0]):
        a_t = mask_invalid(a[t], scl[t], invalid_scl)
        b_t = mask_invalid(b[t], scl[t], invalid_scl)
        per_scene.append(index_func(a_t, b_t))
    stack = np.stack(per_scene, axis=0)
    return temporal_composite(stack, agg="median", axis=0)


def build_change_inputs(
    bbox: list[float],
    baseline_years: tuple[str, str],
    recent_years: tuple[str, str],
    index: str = "ndvi",
    max_cloud: float = 40.0,
    resolution: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (before, after) composites the change detector consumes.

    Loads a cloud-robust median composite of ``index`` for the baseline period
    and for the recent period and returns them as a 2-D numpy pair, ready for
    :func:`eets.change.change_map`.

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    baseline_years, recent_years:
        ``(start_iso, end_iso)`` date ranges for the two periods.
    index:
        ``"ndvi"``, ``"ndwi"``, or ``"nbr"``.
    max_cloud, resolution:
        Passed through to :func:`load_s2_period`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(before_composite, after_composite)``.
    """
    before = load_s2_period(
        bbox, baseline_years[0], baseline_years[1], max_cloud, index, resolution
    )
    after = load_s2_period(
        bbox, recent_years[0], recent_years[1], max_cloud, index, resolution
    )
    return before, after


def aoi_to_bbox(cfg: dict[str, Any]) -> list[float]:
    """Extract an EPSG:4326 ``[min_lon, min_lat, max_lon, max_lat]`` from config."""
    b = cfg["aoi"]["bbox"]
    return [b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"]]
