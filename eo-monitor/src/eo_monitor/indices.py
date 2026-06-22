"""Pure vegetation / moisture index math.

All functions operate elementwise on array-likes (numpy arrays or xarray
DataArrays) and use the standard normalised-difference definitions so the unit
tests can assert exact, hand-checked values.

Index definitions
-----------------
NDVI = (NIR  - Red)  / (NIR  + Red)
NDWI = (Green - NIR) / (Green + NIR)   (McFeeters 1996, open-water/wetness)
NDMI = (NIR  - SWIR) / (NIR  + SWIR)   (vegetation moisture, a.k.a. NDII)

Sentinel-2 L2A band mapping (Earth Search common-name assets)
-------------------------------------------------------------
Red = red (B04), Green = green (B03), NIR = nir (B08), SWIR = swir16 (B11).
The Earth Search catalogue names its assets with these common names rather than
the B-number band IDs, so those are the keys requested from the STAC items.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

# Works for both numpy ndarrays and xarray DataArrays (duck-typed arithmetic).
ArrayT = TypeVar("ArrayT")

# Earth Search (sentinel-2-l2a) asset keys for each logical band used here.
BAND_ALIASES: dict[str, str] = {
    "red": "red",
    "green": "green",
    "nir": "nir",
    "swir": "swir16",
    "swir2": "swir22",
}


def _normalized_difference(a: ArrayT, b: ArrayT) -> ArrayT:
    """Return (a - b) / (a + b) with divide-by-zero handled as NaN.

    The numpy path suppresses warnings and substitutes NaN where a + b == 0.
    xarray DataArrays propagate NaN natively through the same expression.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr = np.asarray(a, dtype="float64")
        b_arr = np.asarray(b, dtype="float64")
        denom = a_arr + b_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom == 0, np.nan, (a_arr - b_arr) / denom)
        return out  # type: ignore[return-value]
    # xarray (or other duck-typed) path: lazy / Dask friendly, no .compute().
    return (a - b) / (a + b)  # type: ignore[operator]


def normalized_difference(a: ArrayT, b: ArrayT) -> ArrayT:
    """Public normalised difference ``(a - b) / (a + b)``.

    The building block behind every band-ratio index here. Divide-by-zero
    (``a + b == 0``) yields NaN rather than raising, and NaN inputs propagate.
    """
    return _normalized_difference(a, b)


def ndvi(nir: ArrayT, red: ArrayT) -> ArrayT:
    """Normalised Difference Vegetation Index = (NIR - Red) / (NIR + Red)."""
    return _normalized_difference(nir, red)


def ndwi(green: ArrayT, nir: ArrayT) -> ArrayT:
    """Normalised Difference Water Index = (Green - NIR) / (Green + NIR)."""
    return _normalized_difference(green, nir)


def ndmi(nir: ArrayT, swir: ArrayT) -> ArrayT:
    """Normalised Difference Moisture Index = (NIR - SWIR) / (NIR + SWIR)."""
    return _normalized_difference(nir, swir)


def savi(nir: ArrayT, red: ArrayT, L: float = 0.5) -> ArrayT:
    """Soil-Adjusted Vegetation Index (Huete 1988).

    ``SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)``. The soil-brightness
    factor ``L`` (default 0.5 for intermediate canopy) damps the soil-background
    signal; at ``L = 0`` SAVI collapses to NDVI. Divide-by-zero
    (``NIR + Red + L == 0``) yields NaN.
    """
    if isinstance(nir, np.ndarray) or isinstance(red, np.ndarray):
        n = np.asarray(nir, dtype="float64")
        r = np.asarray(red, dtype="float64")
        denom = n + r + L
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom == 0, np.nan, (1.0 + L) * (n - r) / denom)
        return out  # type: ignore[return-value]
    return (1.0 + L) * (nir - red) / (nir + red + L)  # type: ignore[operator]


def evi2(nir: ArrayT, red: ArrayT) -> ArrayT:
    """Two-band Enhanced Vegetation Index (Jiang et al. 2008).

    ``EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)``. A blue-band-free EVI
    that keeps EVI's reduced saturation over dense canopy. Divide-by-zero yields
    NaN.
    """
    if isinstance(nir, np.ndarray) or isinstance(red, np.ndarray):
        n = np.asarray(nir, dtype="float64")
        r = np.asarray(red, dtype="float64")
        denom = n + 2.4 * r + 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom == 0, np.nan, 2.5 * (n - r) / denom)
        return out  # type: ignore[return-value]
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)  # type: ignore[operator]


def nbr(nir: ArrayT, swir: ArrayT) -> ArrayT:
    """Normalised Burn Ratio = (NIR - SWIR) / (NIR + SWIR).

    Same normalised-difference form as NDMI but conventionally paired with the
    longer SWIR (S2 B12, swir22); used to map burn severity. Divide-by-zero
    yields NaN.
    """
    return _normalized_difference(nir, swir)


def compute_index(name: str, bands: dict[str, ArrayT]) -> ArrayT:
    """Dispatch by index name using a dict of logical bands.

    Parameters
    ----------
    name
        One of NDVI, NDWI, NDMI, SAVI, EVI2, NBR (case-insensitive).
    bands
        Mapping with the logical band names required by the index, e.g.
        ``{"nir": ..., "red": ...}``. Keys are the lower-case names in
        :data:`BAND_ALIASES`.
    """
    key = name.upper()
    if key == "NDVI":
        return ndvi(bands["nir"], bands["red"])
    if key == "NDWI":
        return ndwi(bands["green"], bands["nir"])
    if key == "NDMI":
        return ndmi(bands["nir"], bands["swir"])
    if key == "SAVI":
        return savi(bands["nir"], bands["red"])
    if key == "EVI2":
        return evi2(bands["nir"], bands["red"])
    if key == "NBR":
        return nbr(bands["nir"], bands["swir2"])
    raise ValueError(f"Unknown index: {name!r}. Supported: NDVI, NDWI, NDMI, SAVI, EVI2, NBR.")


def required_bands(indices: list[str]) -> list[str]:
    """Return the sorted set of Sentinel-2 asset keys needed for given indices."""
    needed: set[str] = set()
    for name in indices:
        key = name.upper()
        if key == "NDVI":
            needed.update({"red", "nir"})
        elif key == "NDWI":
            needed.update({"green", "nir"})
        elif key == "NDMI":
            needed.update({"nir", "swir"})
        elif key in ("SAVI", "EVI2"):
            needed.update({"red", "nir"})
        elif key == "NBR":
            needed.update({"nir", "swir2"})
        else:
            raise ValueError(f"Unknown index: {name!r}.")
    return sorted(BAND_ALIASES[b] for b in needed)
