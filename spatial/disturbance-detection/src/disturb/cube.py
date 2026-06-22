"""Build an analysis-ready NDVI time cube from a STAC catalogue.

The cube-building approach (cloud-native, Dask-backed ``xarray``) is shared
with the **eo-monitor** project, applied here to the *time* axis: a dense,
regularly-spaced NDVI series per pixel over several years for downstream
harmonic decomposition and breakpoint detection.

Pipeline
--------
1. Search a STAC API (Planetary Computer HLS, or Landsat C2 L2) for items over
   the AOI and date range.
2. ``odc.stac.load`` the required bands lazily as a Dask-backed cube.
3. Apply the cloud/shadow mask from the scene QA band; masked pixels become
   **NaN, never 0** (a zero would be read as a real low-NDVI observation and
   create phantom disturbances).
4. Compute NDVI = (NIR - Red) / (NIR + Red).
5. Resample onto a *regular* time index (e.g. 16-day medians) leaving genuine
   gaps as NaN.

All geospatial dependencies are imported lazily so that ``import disturb.cube``
never fails on a machine without the EO stack installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Band aliases per collection. NDVI needs red + nir; QA carries the cloud mask.
_BAND_PRESETS: dict[str, dict[str, str]] = {
    # HLS (HLSS30 / HLSL30) via Microsoft Planetary Computer.
    "hls": {"red": "B04", "nir": "B8A", "qa": "Fmask"},
    # Landsat Collection-2 Level-2 surface reflectance.
    "landsat-c2-l2": {"red": "red", "nir": "nir08", "qa": "qa_pixel"},
}

_DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _import_eo_stack():
    """Lazily import the heavy EO stack with a friendly error message."""
    try:
        import odc.stac  # noqa: F401
        import planetary_computer as pc
        import pystac_client
        import xarray as xr  # noqa: F401

        return odc.stac, pc, pystac_client, xr
    except ImportError as exc:  # pragma: no cover - needs full geo stack
        raise ImportError(
            "Building a cube requires the geospatial stack "
            "(odc-stac, pystac-client, planetary-computer, xarray, rioxarray, "
            "dask). Install via `pixi install` or `pip install -r "
            "requirements.txt`."
        ) from exc


def search_items(
    bbox: list[float],
    start: str,
    end: str,
    collection: str = "hls",
    stac_url: str = _DEFAULT_STAC_URL,
    query: dict[str, Any] | None = None,
) -> list:
    """Search a STAC API and return signed items over the AOI/date range.

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    start, end:
        ISO dates, e.g. ``"2018-01-01"``, ``"2023-12-31"``.
    collection:
        ``"hls"`` (mapped to HLSS30/HLSL30) or ``"landsat-c2-l2"``.
    stac_url:
        STAC API root (default: Planetary Computer).
    query:
        Optional STAC ``query`` filter (e.g. cloud-cover bounds).
    """
    _, pc, pystac_client, _ = _import_eo_stack()

    collections = (
        ["hls2-s30", "hls2-l30"] if collection == "hls" else ["landsat-c2-l2"]
    )
    catalog = pystac_client.Client.open(
        stac_url, modifier=pc.sign_inplace
    )
    search = catalog.search(
        collections=collections,
        bbox=bbox,
        datetime=f"{start}/{end}",
        query=query or {},
    )
    return list(search.items())


def _cloud_mask(qa, collection: str):
    """Boolean mask of *clear* pixels from the collection's QA band."""
    if collection == "hls":
        # HLS Fmask bits: 1=cloud, 2=adjacent, 3=shadow. Keep where none set.
        cloud = (qa.astype("uint16") & 0b0000_1110) != 0
        return ~cloud
    # Landsat C2 qa_pixel bits: 1=dilated cloud, 3=cloud, 4=shadow.
    bits = (1 << 1) | (1 << 3) | (1 << 4)
    cloud = (qa.astype("uint16") & bits) != 0
    return ~cloud


def build_ndvi_cube(
    bbox: list[float],
    start: str,
    end: str,
    collection: str = "hls",
    resolution: float = 30.0,
    freq: str = "16D",
    chunks: dict[str, int] | None = None,
    stac_url: str = _DEFAULT_STAC_URL,
):
    """Build a masked, regularly-sampled NDVI time cube.

    Returns a Dask-backed ``xarray.DataArray`` with dims ``(time, y, x)`` where
    cloudy / missing observations are ``NaN`` (never 0) and the time axis is
    resampled to ``freq`` (default 16-day median composites).

    Parameters
    ----------
    bbox, start, end, collection, stac_url:
        See :func:`search_items`.
    resolution:
        Output pixel size in metres.
    freq:
        Pandas offset alias for the regular time grid (e.g. ``"16D"``, ``"M"``).
    chunks:
        Dask chunking, default ``{"time": 1, "x": 1024, "y": 1024}``.
    """
    odc_stac, _, _, xr = _import_eo_stack()

    if collection not in _BAND_PRESETS:
        raise ValueError(f"unknown collection: {collection!r}")
    bands = _BAND_PRESETS[collection]
    chunks = chunks or {"time": 1, "x": 1024, "y": 1024}

    items = search_items(bbox, start, end, collection, stac_url)
    if not items:
        raise RuntimeError("no STAC items found for AOI/date range")

    ds = odc_stac.load(
        items,
        bands=[bands["red"], bands["nir"], bands["qa"]],
        bbox=bbox,
        resolution=resolution,
        chunks=chunks,
        groupby="solar_day",
    )

    red = ds[bands["red"]].astype("float32")
    nir = ds[bands["nir"]].astype("float32")
    keep = _cloud_mask(ds[bands["qa"]], collection)

    # Mask BEFORE computing NDVI; masked pixels are NaN, never 0.
    red = red.where(keep)
    nir = nir.where(keep)

    denom = nir + red
    ndvi = (nir - red) / denom
    # Guard division by zero (denom == 0 -> inf) -> NaN, then clip valid range.
    ndvi = ndvi.where(np.isfinite(ndvi))
    ndvi = ndvi.clip(-1.0, 1.0)
    ndvi.name = "ndvi"

    # Resample to a regular time index; gaps stay NaN (skipna median).
    ndvi = ndvi.resample(time=freq).median(skipna=True)

    ndvi.attrs.update(
        {
            "long_name": "Normalized Difference Vegetation Index",
            "collection": collection,
            "resample_freq": freq,
            "masking": "cloud/shadow -> NaN (never 0)",
        }
    )
    return ndvi
