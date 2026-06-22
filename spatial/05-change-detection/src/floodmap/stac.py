"""Default real-data path: Sentinel-1 GRD (and S2 fallback) from Earth Search.

This is the **auth-free** way to feed the pure-numpy flood core with real
imagery. It queries the Element-84 **Earth Search** STAC API for Sentinel-1 GRD
items (radar, sees through cloud) over an AOI and a date, loads the requested
polarization with ``odc.stac``, and returns the backscatter array. A small
Sentinel-2 helper returns an MNDWI image for the cloud-free-day optical fallback.
No account, no Google sign-in, no API key.

The heavy geospatial dependencies (``pystac-client``, ``odc-stac`` / ``odc-geo``,
``rioxarray``, ``xarray``, ``rasterio``) are imported lazily *inside* the
functions, so ``import floodmap.stac`` never fails on a machine that only has
numpy, and the test suite never imports this module.

Earth Search collections / assets used:

* ``sentinel-1-grd`` — assets ``vv`` and ``vh`` (linear-amplitude backscatter).
  ``VH`` is recommended for open-water mapping (low cross-pol return over smooth
  water gives strong land/water contrast).
* ``sentinel-2-l2a`` — assets ``green`` (B03) and ``swir16`` (B11) for MNDWI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from floodmap.water import mndwi, to_db

# Element-84 Earth Search v1 — open, no authentication.
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S1_COLLECTION = "sentinel-1-grd"
S2_COLLECTION = "sentinel-2-l2a"


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


def load_s1_scene(
    bbox: list[float],
    date: str,
    orbit: str = "descending",
    polarization: str = "vh",
    resolution: float = 10.0,
    to_decibels: bool = True,
) -> np.ndarray:
    """Load a Sentinel-1 GRD backscatter scene for a date from Earth Search.

    Searches Earth Search for Sentinel-1 GRD items over ``bbox`` on ``date``
    (a single day or an ISO ``start/end`` range) for the requested orbit
    direction, loads the chosen ``polarization`` band, mosaics same-day passes,
    and returns the 2-D backscatter array — converted to decibels by default
    (the scale Otsu thresholding expects).

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    date:
        ISO date ``"2022-09-15"`` or range ``"2022-09-10/2022-09-20"``.
    orbit:
        ``"descending"`` or ``"ascending"``; keep one orbit so the incidence
        angle is consistent between the pre and post scenes.
    polarization:
        ``"vh"`` (recommended for water) or ``"vv"``.
    resolution:
        Output pixel size in metres (10 for S1 GRD).
    to_decibels:
        If ``True`` (default), return ``10*log10`` backscatter in dB.

    Returns
    -------
    numpy.ndarray
        2-D backscatter array (dB if ``to_decibels``), for Otsu + water masking.
    """
    pol = polarization.lower()
    if pol not in ("vv", "vh"):
        raise ValueError(f"polarization must be 'vv' or 'vh', got {polarization!r}")
    odc_stac, pystac_client, _ = _import_eo_stack()

    datetime = date if "/" in date else f"{date}/{date}"
    catalog = pystac_client.Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=[S1_COLLECTION],
        bbox=bbox,
        datetime=datetime,
        query={"sat:orbit_state": {"eq": orbit}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("no Sentinel-1 GRD items found for AOI / date / orbit")

    ds = odc_stac.load(
        items,
        bands=[pol],
        bbox=bbox,
        resolution=resolution,
        groupby="solar_day",
        chunks={},
    )
    # Mosaic any same-day passes to a single 2-D scene (mean over time).
    linear = ds[pol].astype("float32").mean(dim="time").to_numpy()
    return to_db(linear) if to_decibels else linear


def build_flood_inputs(
    bbox: list[float],
    pre_date: str,
    post_date: str,
    orbit: str = "descending",
    polarization: str = "vh",
    resolution: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (pre, post) backscatter pair the flood mapper consumes.

    Loads a Sentinel-1 GRD backscatter scene (in dB) for ``pre_date`` and for
    ``post_date`` over the same AOI / orbit / polarization, returning them as a
    2-D numpy pair, ready for :func:`floodmap.threshold.otsu_threshold` and
    :func:`floodmap.change.flood_extent`.

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    pre_date, post_date:
        ISO dates (or ranges) for the pre-flood and post-flood scenes.
    orbit, polarization, resolution:
        Passed through to :func:`load_s1_scene`. Keep orbit and polarization the
        same for both dates so the backscatter is comparable.

    Returns
    -------
    tuple of numpy.ndarray
        ``(pre_backscatter_db, post_backscatter_db)``.
    """
    pre = load_s1_scene(bbox, pre_date, orbit, polarization, resolution)
    post = load_s1_scene(bbox, post_date, orbit, polarization, resolution)
    return pre, post


def load_s2_mndwi(bbox: list[float], date: str, resolution: float = 10.0) -> np.ndarray:
    """Optical fallback: an MNDWI image for a date from Sentinel-2 L2A.

    For the cloud-free-day fallback path. Loads the Green (B03) and SWIR16 (B11)
    bands over the AOI and returns the MNDWI image (water is high MNDWI, so use
    ``water_mask(..., polarity="above")``). Optical imagery is cloud-blocked,
    which is exactly why SAR is the default flood path.

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    date:
        ISO date or ``start/end`` range.
    resolution:
        Output pixel size in metres.

    Returns
    -------
    numpy.ndarray
        2-D MNDWI array.
    """
    odc_stac, pystac_client, _ = _import_eo_stack()

    datetime = date if "/" in date else f"{date}/{date}"
    catalog = pystac_client.Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": 60}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("no Sentinel-2 items found for AOI / date")

    ds = odc_stac.load(
        items,
        bands=["green", "swir16"],
        bbox=bbox,
        resolution=resolution,
        groupby="solar_day",
        chunks={},
    )
    green = ds["green"].astype("float32").median(dim="time").to_numpy()
    swir = ds["swir16"].astype("float32").median(dim="time").to_numpy()
    return mndwi(green, swir)


def aoi_to_bbox(cfg: dict[str, Any]) -> list[float]:
    """Extract an EPSG:4326 ``[min_lon, min_lat, max_lon, max_lat]`` from config."""
    b = cfg["aoi"]["bbox"]
    return [b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"]]
