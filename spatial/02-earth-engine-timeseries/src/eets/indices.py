"""Pure-numpy spectral indices for Sentinel-2 surface reflectance.

These are the normalized-difference indices the time-series and change-detection
code is built on. They take numpy arrays of surface reflectance (any shape,
broadcastable) and return arrays the same shape. Division by zero — a pixel
where both bands are zero, e.g. a masked nodata pixel — yields ``NaN`` rather
than ``inf`` or a misleading 0, so masked pixels never read as real low/high
index values downstream.

Index conventions (Sentinel-2 band roles):

* NDVI = (NIR - Red) / (NIR + Red)            — vegetation greenness
* NDWI = (Green - NIR) / (Green + NIR)        — open water / moisture (McFeeters)
* NBR  = (NIR - SWIR) / (NIR + SWIR)          — burn / live-vegetation contrast

Everything here depends on numpy and the standard library only.
"""

from __future__ import annotations

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the normalized difference ``(a - b) / (a + b)``.

    Computed in floating point. Where the denominator ``a + b`` is zero the
    result is ``NaN`` (not ``inf``), so nodata / fully-masked pixels do not
    masquerade as a valid index value.

    Parameters
    ----------
    a, b:
        Reflectance arrays, broadcastable to a common shape.

    Returns
    -------
    numpy.ndarray
        Float array of the normalized difference, ``NaN`` where ``a + b == 0``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = a - b
    den = a + b
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index, ``(NIR - Red) / (NIR + Red)``."""
    return normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters Normalized Difference Water Index, ``(Green - NIR)/(Green+NIR)``."""
    return normalized_difference(green, nir)


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio, ``(NIR - SWIR) / (NIR + SWIR)``."""
    return normalized_difference(nir, swir)
