"""Optional Earth Engine path: the geemap / earthengine-api equivalent.

This module mirrors the default STAC pipeline (:mod:`eets.stac`) on **Google
Earth Engine**, included for parity with the project that inspired this one
(`giswqs/geemap`). It is **not** required to run the project: the auth-free
Earth Search / STAC path is the default tested route. Use this only when you
have authenticated Earth Engine and want server-side compositing.

Setup (one-time, free Google account)::

    pip install earthengine-api geemap
    earthengine authenticate          # opens a browser, free Google sign-in
    # then in code: ee.Initialize(project="your-cloud-project-id")

``earthengine-api`` / ``geemap`` are imported lazily inside the functions, so
``import eets.gee`` never fails without them and the test suite never touches
this module. The same NDVI -> composite -> change -> hectares logic runs as Earth
Engine server-side operations on ``COPERNICUS/S2_SR_HARMONIZED``.
"""

from __future__ import annotations

from typing import Any

# Earth Engine Sentinel-2 SR band names (note: distinct from Earth Search keys).
#   B8 = NIR, B4 = Red, B3 = Green, B11 = SWIR1, B12 = SWIR2, SCL = scene class.
S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
_INDEX_BANDS = {
    "ndvi": ("B8", "B4"),
    "ndwi": ("B3", "B8"),
    "nbr": ("B8", "B11"),
}
# SCL cloud/shadow/snow classes to drop (same semantics as the STAC path).
_INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)


def _import_ee():
    """Lazily import earthengine-api with a friendly error message."""
    try:
        import ee  # noqa: F401

        return ee
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError(
            "The Earth Engine path needs earthengine-api + geemap and a "
            "one-time `earthengine authenticate`. This is optional; the default "
            "STAC path (eets.stac) is auth-free."
        ) from exc


def initialize(project: str | None = None) -> None:
    """Initialise Earth Engine. Requires a prior ``earthengine authenticate``."""
    ee = _import_ee()
    ee.Initialize(project=project)


def _mask_and_index(image, index: str):
    """Cloud-mask one S2_SR image via SCL and return the requested index band."""
    scl = image.select("SCL")
    keep = scl.neq(_INVALID_SCL[0])
    for cls in _INVALID_SCL[1:]:
        keep = keep.And(scl.neq(cls))
    band_a, band_b = _INDEX_BANDS[index]
    idx = (
        image.select(band_a)
        .subtract(image.select(band_b))
        .divide(image.select(band_a).add(image.select(band_b)))
        .updateMask(keep)
        .rename(index)
    )
    return idx


def period_composite(
    bbox: list[float],
    start: str,
    end: str,
    index: str = "ndvi",
    max_cloud: float = 40.0,
):
    """Median index composite for a period as an ``ee.Image`` (server-side).

    Parameters
    ----------
    bbox:
        ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    start, end:
        ISO dates.
    index:
        ``"ndvi"``, ``"ndwi"``, or ``"nbr"``.
    max_cloud:
        Maximum scene cloud cover percentage.
    """
    ee = _import_ee()
    if index not in _INDEX_BANDS:
        raise ValueError(f"unknown index {index!r}")
    region = ee.Geometry.Rectangle(bbox)
    collection = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(lambda img: _mask_and_index(img, index))
    )
    return collection.median().clip(region)


def change_composite(
    bbox: list[float],
    baseline_years: tuple[str, str],
    recent_years: tuple[str, str],
    index: str = "ndvi",
    max_cloud: float = 40.0,
) -> dict[str, Any]:
    """Return baseline, recent, and ``recent - baseline`` change as ee.Images.

    The Earth Engine equivalent of :func:`eets.stac.build_change_inputs`. Returns
    a dict with ``before``, ``after`` and ``change`` ``ee.Image`` objects; the
    caller can threshold / reduceRegion to hectares on the server.
    """
    before = period_composite(bbox, *baseline_years, index, max_cloud)
    after = period_composite(bbox, *recent_years, index, max_cloud)
    return {"before": before, "after": after, "change": after.subtract(before)}
