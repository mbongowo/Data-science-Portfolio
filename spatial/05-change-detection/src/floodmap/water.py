"""Pure-numpy water detection: SAR backscatter and an optical fallback.

Two ways to find water, plus the SAR unit conversion that makes Otsu work:

* :func:`water_mask` turns a thresholded image into a boolean water mask. For
  Sentinel-1 SAR backscatter water is *dark* (low backscatter), so the default
  ``polarity="below"`` marks pixels **below** the threshold as water. For an
  optical water index (MNDWI) water is *bright*, so ``polarity="above"`` marks
  pixels above the threshold.
* :func:`to_db` converts linear SAR backscatter to decibels. Otsu on dB is the
  standard recipe: the log compresses the long bright tail of land/urban
  backscatter so the water and land peaks become cleanly separable.
* :func:`mndwi` is the optical Modified Normalized Difference Water Index, the
  cloud-free-day fallback when SAR is unavailable.

NaN is the missing-data sentinel throughout: a NaN pixel is never water, and the
index / dB conversions yield NaN rather than a misleading number. Everything here
depends on numpy and the standard library only.
"""

from __future__ import annotations

import numpy as np


def water_mask(
    image: np.ndarray, threshold: float, polarity: str = "below"
) -> np.ndarray:
    """Return a boolean water mask from a thresholded image.

    Parameters
    ----------
    image:
        The scalar image being thresholded — SAR backscatter in dB, or an optical
        water index. ``NaN`` pixels are never water.
    threshold:
        The water/non-water cut-off, e.g. from
        :func:`floodmap.threshold.otsu_threshold`.
    polarity:
        ``"below"`` (default) marks ``image < threshold`` as water — correct for
        SAR backscatter, where water is dark. ``"above"`` marks
        ``image > threshold`` as water — correct for an optical water index like
        MNDWI, where water is bright.

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` where the pixel is water. ``NaN`` -> ``False``.

    Raises
    ------
    ValueError
        If ``polarity`` is not ``"below"`` or ``"above"``.
    """
    if polarity not in ("below", "above"):
        raise ValueError(f"polarity must be 'below' or 'above', got {polarity!r}")
    arr = np.asarray(image, dtype=np.float64)
    valid = ~np.isnan(arr)
    if polarity == "below":
        return valid & (arr < threshold)
    return valid & (arr > threshold)


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified Normalized Difference Water Index, ``(Green - SWIR)/(Green + SWIR)``.

    The optical water index (Xu 2006), used as the cloud-free-day fallback to the
    SAR path. Open water is **high** MNDWI (use ``polarity="above"`` in
    :func:`water_mask`). Division by zero — both bands zero, e.g. a masked nodata
    pixel — yields ``NaN`` rather than ``inf``.

    Parameters
    ----------
    green, swir:
        Reflectance arrays (Sentinel-2 Green / SWIR), broadcastable.

    Returns
    -------
    numpy.ndarray
        Float MNDWI array, ``NaN`` where ``Green + SWIR == 0``.
    """
    g = np.asarray(green, dtype=np.float64)
    s = np.asarray(swir, dtype=np.float64)
    num = g - s
    den = g + s
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear SAR backscatter to decibels, ``10 * log10(linear)``.

    Sentinel-1 GRD backscatter (sigma-nought) is delivered as a linear power
    ratio; Otsu thresholding is done on the dB scale, where the histogram is
    cleanly bimodal. Non-positive inputs (``<= 0``) have no real logarithm and
    become ``NaN`` (so e.g. ``to_db(0.1) == -10.0``).

    Parameters
    ----------
    linear:
        Linear backscatter values (any shape).

    Returns
    -------
    numpy.ndarray
        Backscatter in decibels, ``NaN`` where ``linear <= 0``.
    """
    arr = np.asarray(linear, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    positive = arr > 0
    out[positive] = 10.0 * np.log10(arr[positive])
    return out
