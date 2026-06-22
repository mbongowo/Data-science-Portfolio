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
# The logical names on the left are what the index functions and ``compute_index``
# speak; the values are the actual Earth Search common-name assets to request.
BAND_ALIASES: dict[str, str] = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "nir": "nir",
    "rededge": "rededge1",  # "RE" / red-edge -> Sentinel-2 B05
    "rededge1": "rededge1",
    "swir": "swir16",  # SWIR1 (B11)
    "swir1": "swir16",
    "swir16": "swir16",
    "swir2": "swir22",  # SWIR2 (B12)
    "swir22": "swir22",
}


def _to_f64(*arrays: ArrayT):
    """Cast every argument to a float64 numpy array (helper for scaled indices)."""
    return tuple(np.asarray(a, dtype="float64") for a in arrays)


def _safe_divide(num, denom):
    """Return ``num / denom`` with divide-by-zero -> NaN (numpy path).

    Used by ratio and additive-constant indices that are not a plain normalised
    difference. NaN inputs propagate; ``denom == 0`` becomes NaN rather than
    raising or producing +/-inf.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, num / denom)
    return out


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


# --------------------------------------------------------------------------- #
# Vegetation indices
# --------------------------------------------------------------------------- #
#
# Reflectance note: normalised-difference and pure-ratio indices below are
# scale-invariant (a constant DN scale cancels in numerator and denominator),
# so they work on raw DN or reflectance. Indices with ADDITIVE CONSTANTS (EVI,
# SAVI, MSAVI, AWEI, BAI) assume surface reflectance in [0, 1]; feed them scaled
# Sentinel-2 L2A reflectance, not raw DN. The app's ``stac.load_scene`` applies
# the DN * 1e-4 (+ baseline-04.00 -0.1 offset) scaling before computing indices.


def evi(nir: ArrayT, red: ArrayT, blue: ArrayT) -> ArrayT:
    """Enhanced Vegetation Index (Huete et al. 2002).

    ``EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)``. Uses the blue
    band to correct for aerosol scattering and reduces canopy saturation.
    **Assumes reflectance in [0, 1]** because of the additive ``+ 1``.
    Divide-by-zero -> NaN.
    """
    n, r, b = _to_f64(nir, red, blue)
    return _safe_divide(2.5 * (n - r), n + 6.0 * r - 7.5 * b + 1.0)


def msavi(nir: ArrayT, red: ArrayT) -> ArrayT:
    """Modified Soil-Adjusted Vegetation Index (Qi et al. 1994).

    ``MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2``. A
    self-adjusting SAVI whose soil factor is computed from the data rather than
    fixed. **Assumes reflectance in [0, 1]** (additive constants). NaN where the
    radicand is negative.
    """
    n, r = _to_f64(nir, red)
    radicand = (2.0 * n + 1.0) ** 2 - 8.0 * (n - r)
    with np.errstate(invalid="ignore"):
        root = np.sqrt(np.where(radicand < 0, np.nan, radicand))
    return (2.0 * n + 1.0 - root) / 2.0


def gndvi(nir: ArrayT, green: ArrayT) -> ArrayT:
    """Green NDVI = (NIR - Green) / (NIR + Green) (Gitelson et al. 1996)."""
    return _normalized_difference(nir, green)


def arvi(nir: ArrayT, red: ArrayT, blue: ArrayT) -> ArrayT:
    """Atmospherically Resistant Vegetation Index (Kaufman & Tanre 1992).

    ``ARVI = (NIR - rb) / (NIR + rb)`` with ``rb = 2*Red - Blue``; the blue band
    self-corrects red for atmospheric scattering. Divide-by-zero -> NaN.
    """
    n, r, b = _to_f64(nir, red, blue)
    rb = 2.0 * r - b
    return _normalized_difference(n, rb)


def ndre(nir: ArrayT, rededge: ArrayT) -> ArrayT:
    """Normalised Difference Red-Edge = (NIR - RE) / (NIR + RE).

    Uses the red-edge band (S2 B05) instead of red; sensitive to chlorophyll in
    moderate-to-dense canopy where NDVI saturates.
    """
    return _normalized_difference(nir, rededge)


def vari(green: ArrayT, red: ArrayT, blue: ArrayT) -> ArrayT:
    """Visible Atmospherically Resistant Index (Gitelson et al. 2002).

    ``VARI = (Green - Red) / (Green + Red - Blue)``. A visible-band greenness
    index for RGB sensors. Divide-by-zero -> NaN.
    """
    g, r, b = _to_f64(green, red, blue)
    return _safe_divide(g - r, g + r - b)


def rvi(nir: ArrayT, red: ArrayT) -> ArrayT:
    """Ratio Vegetation Index = NIR / Red (Jordan 1969). Red == 0 -> NaN."""
    n, r = _to_f64(nir, red)
    return _safe_divide(n, r)


def dvi(nir: ArrayT, red: ArrayT) -> ArrayT:
    """Difference Vegetation Index = NIR - Red (Tucker 1979)."""
    n, r = _to_f64(nir, red)
    return n - r


def ci_green(nir: ArrayT, green: ArrayT) -> ArrayT:
    """Chlorophyll Index - green = NIR / Green - 1 (Gitelson et al. 2003).

    Green == 0 -> NaN.
    """
    n, g = _to_f64(nir, green)
    return _safe_divide(n, g) - 1.0


def ci_rededge(nir: ArrayT, rededge: ArrayT) -> ArrayT:
    """Chlorophyll Index - red-edge = NIR / RE - 1 (Gitelson et al. 2003).

    RE == 0 -> NaN.
    """
    n, re = _to_f64(nir, rededge)
    return _safe_divide(n, re) - 1.0


def mcari(rededge: ArrayT, red: ArrayT, green: ArrayT) -> ArrayT:
    """Modified Chlorophyll Absorption in Reflectance Index (Daughtry 2000).

    ``MCARI = ((RE - Red) - 0.2*(RE - Green)) * (RE / Red)``. Red == 0 -> NaN.
    """
    re, r, g = _to_f64(rededge, red, green)
    return ((re - r) - 0.2 * (re - g)) * _safe_divide(re, r)


def tcari(rededge: ArrayT, red: ArrayT, green: ArrayT) -> ArrayT:
    """Transformed Chlorophyll Absorption in Reflectance Index (Haboudane 2002).

    ``TCARI = 3 * ((RE - Red) - 0.2*(RE - Green) * (RE / Red))``. Red == 0 -> NaN.
    """
    re, r, g = _to_f64(rededge, red, green)
    return 3.0 * ((re - r) - 0.2 * (re - g) * _safe_divide(re, r))


def lai(nir: ArrayT, red: ArrayT, blue: ArrayT) -> ArrayT:
    """Leaf Area Index, approximate empirical relation from EVI.

    ``LAI = 3.618 * EVI - 0.118`` (Boegh et al. 2002). This is an **approximate,
    empirical** crop-calibrated relation, not a physically retrieved LAI; treat
    the numbers as relative. Inherits EVI's reflectance-in-[0,1] assumption.
    """
    return 3.618 * evi(nir, red, blue) - 0.118


# --------------------------------------------------------------------------- #
# Water & moisture indices
# --------------------------------------------------------------------------- #


def mndwi(green: ArrayT, swir1: ArrayT) -> ArrayT:
    """Modified NDWI = (Green - SWIR1) / (Green + SWIR1) (Xu 2006).

    Uses SWIR instead of NIR; suppresses built-up land better than NDWI for
    open-water mapping.
    """
    return _normalized_difference(green, swir1)


def awei(green: ArrayT, nir: ArrayT, swir1: ArrayT, swir2: ArrayT) -> ArrayT:
    """Automated Water Extraction Index, no-shadow form (Feyisa et al. 2014).

    ``AWEI_nsh = 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)``. **Assumes
    reflectance in [0, 1]** because of the fixed coefficients. Positive over
    water.
    """
    g, n, s1, s2 = _to_f64(green, nir, swir1, swir2)
    return 4.0 * (g - s1) - (0.25 * n + 2.75 * s2)


def ndii(nir: ArrayT, swir1: ArrayT) -> ArrayT:
    """Normalised Difference Infrared Index = (NIR - SWIR1) / (NIR + SWIR1).

    Identical formula to :func:`ndmi` (Hardisky et al. 1983); kept under its own
    name because the literature uses both. Sensitive to canopy water content.
    """
    return _normalized_difference(nir, swir1)


# --------------------------------------------------------------------------- #
# Soil & geology indices
# --------------------------------------------------------------------------- #


def bsi(swir1: ArrayT, red: ArrayT, nir: ArrayT, blue: ArrayT) -> ArrayT:
    """Bare Soil Index (Rikimaru et al. 2002).

    ``BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))``.
    Normalised-difference form, so scale-invariant. Divide-by-zero -> NaN.
    """
    s1, r, n, b = _to_f64(swir1, red, nir, blue)
    return _normalized_difference(s1 + r, n + b)


def salinity_index(green: ArrayT, red: ArrayT) -> ArrayT:
    """Soil Salinity Index = sqrt(Green * Red) (one common form).

    Several salinity indices exist; this is the simple ``sqrt(Green*Red)``
    brightness form (Khan et al. 2005). NaN where ``Green*Red < 0``.
    """
    g, r = _to_f64(green, red)
    prod = g * r
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.where(prod < 0, np.nan, prod))


def iron_oxide(red: ArrayT, blue: ArrayT) -> ArrayT:
    """Iron Oxide ratio = Red / Blue (Segal 1982). Blue == 0 -> NaN."""
    r, b = _to_f64(red, blue)
    return _safe_divide(r, b)


def clay_minerals(swir1: ArrayT, swir2: ArrayT) -> ArrayT:
    """Clay Minerals ratio = SWIR1 / SWIR2. SWIR2 == 0 -> NaN."""
    s1, s2 = _to_f64(swir1, swir2)
    return _safe_divide(s1, s2)


def ferrous_minerals(swir1: ArrayT, nir: ArrayT) -> ArrayT:
    """Ferrous Minerals ratio = SWIR1 / NIR. NIR == 0 -> NaN."""
    s1, n = _to_f64(swir1, nir)
    return _safe_divide(s1, n)


# --------------------------------------------------------------------------- #
# Built-up / urban indices
# --------------------------------------------------------------------------- #


def ndbi(swir1: ArrayT, nir: ArrayT) -> ArrayT:
    """Normalised Difference Built-up Index = (SWIR1 - NIR) / (SWIR1 + NIR) (Zha 2003)."""
    return _normalized_difference(swir1, nir)


def ui(swir2: ArrayT, nir: ArrayT) -> ArrayT:
    """Urban Index = (SWIR2 - NIR) / (SWIR2 + NIR) (Kawamura 1996)."""
    return _normalized_difference(swir2, nir)


def ibi(swir1: ArrayT, nir: ArrayT, red: ArrayT, green: ArrayT) -> ArrayT:
    """Index-Based Built-up Index (Xu 2008).

    Combines three sub-indices computed internally:
    ``NDBI = (SWIR1-NIR)/(SWIR1+NIR)``, ``SAVI`` (L=0.5, from NIR & Red), and
    ``MNDWI = (Green-SWIR1)/(Green+SWIR1)``, then
    ``IBI = (NDBI - (SAVI+MNDWI)/2) / (NDBI + (SAVI+MNDWI)/2)``.
    **SAVI's reflectance assumption applies** (additive L term). Divide-by-zero -> NaN.
    """
    ndbi_v = ndbi(swir1, nir)
    savi_v = savi(nir, red)
    mndwi_v = mndwi(green, swir1)
    half = (savi_v + mndwi_v) / 2.0
    return _normalized_difference(ndbi_v, half)


# --------------------------------------------------------------------------- #
# Snow / ice indices
# --------------------------------------------------------------------------- #


def ndsi(green: ArrayT, swir1: ArrayT) -> ArrayT:
    """Normalised Difference Snow Index = (Green - SWIR1) / (Green + SWIR1) (Hall 1995).

    Same formula as :func:`mndwi`; the snow/water distinction is by threshold and
    context, not the maths.
    """
    return _normalized_difference(green, swir1)


def ndgi(green: ArrayT, red: ArrayT) -> ArrayT:
    """Normalised Difference Glacier Index = (Green - Red) / (Green + Red).

    One of several glacier-index variants (this is the green/red form); others
    use NIR. Documented here as the green/red definition.
    """
    return _normalized_difference(green, red)


# --------------------------------------------------------------------------- #
# Fire / burn indices
# --------------------------------------------------------------------------- #


def nbr2(swir1: ArrayT, swir2: ArrayT) -> ArrayT:
    """Normalised Burn Ratio 2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2).

    A SWIR-only burn ratio sensitive to post-fire moisture change.
    """
    return _normalized_difference(swir1, swir2)


def bai(red: ArrayT, nir: ArrayT) -> ArrayT:
    """Burned Area Index = 1 / ((0.1 - Red)^2 + (0.06 - NIR)^2) (Chuvieco 2002).

    Highlights charcoal-dark burn scars. **Assumes reflectance in [0, 1]** (the
    0.1 / 0.06 reference values are reflectances). Divide-by-zero (a pixel
    exactly at the reference point) -> NaN.
    """
    r, n = _to_f64(red, nir)
    denom = (0.1 - r) ** 2 + (0.06 - n) ** 2
    return _safe_divide(np.ones_like(denom), denom)


# --------------------------------------------------------------------------- #
# Excluded by design (NOT dispatchable / not in the picker)
# --------------------------------------------------------------------------- #
# EBBI (Enhanced Built-up and Bareness Index) needs a thermal band; Sentinel-2
#   carries no thermal channel, so it cannot be computed here.
# dNBR (differenced NBR) needs two dates (pre- and post-fire NBR). The
#   single-scene index path cannot produce it; it belongs to eo-monitor's
#   temporal/anomaly workflow (see anomaly.py), which already differences a
#   target window against a baseline.


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

#: name -> (function, ordered logical band names matching the function signature).
_INDEX_DISPATCH: dict[str, tuple] = {
    # Vegetation
    "NDVI": (ndvi, ("nir", "red")),
    "EVI": (evi, ("nir", "red", "blue")),
    "EVI2": (evi2, ("nir", "red")),
    "SAVI": (savi, ("nir", "red")),
    "MSAVI": (msavi, ("nir", "red")),
    "GNDVI": (gndvi, ("nir", "green")),
    "ARVI": (arvi, ("nir", "red", "blue")),
    "NDRE": (ndre, ("nir", "rededge")),
    "VARI": (vari, ("green", "red", "blue")),
    "RVI": (rvi, ("nir", "red")),
    "DVI": (dvi, ("nir", "red")),
    "CIGREEN": (ci_green, ("nir", "green")),
    "CIREDEDGE": (ci_rededge, ("nir", "rededge")),
    "MCARI": (mcari, ("rededge", "red", "green")),
    "TCARI": (tcari, ("rededge", "red", "green")),
    "LAI": (lai, ("nir", "red", "blue")),
    # Water & moisture
    "NDWI": (ndwi, ("green", "nir")),
    "MNDWI": (mndwi, ("green", "swir")),
    "NDMI": (ndmi, ("nir", "swir")),
    "AWEI": (awei, ("green", "nir", "swir", "swir2")),
    "NDII": (ndii, ("nir", "swir")),
    # Soil & geology
    "BSI": (bsi, ("swir", "red", "nir", "blue")),
    "SI": (salinity_index, ("green", "red")),
    "IRONOXIDE": (iron_oxide, ("red", "blue")),
    "CLAYMINERALS": (clay_minerals, ("swir", "swir2")),
    "FERROUSMINERALS": (ferrous_minerals, ("swir", "nir")),
    # Built-up / urban
    "NDBI": (ndbi, ("swir", "nir")),
    "UI": (ui, ("swir2", "nir")),
    "IBI": (ibi, ("swir", "nir", "red", "green")),
    # Snow / ice
    "NDSI": (ndsi, ("green", "swir")),
    "NDGI": (ndgi, ("green", "red")),
    # Fire / burn
    "NBR": (nbr, ("nir", "swir2")),
    "NBR2": (nbr2, ("swir", "swir2")),
    "BAI": (bai, ("red", "nir")),
}


def compute_index(name: str, bands: dict[str, ArrayT]) -> ArrayT:
    """Dispatch by index name using a dict of logical bands.

    Parameters
    ----------
    name
        Any index in :data:`_INDEX_DISPATCH` (case-insensitive), e.g. NDVI,
        EVI, SAVI, MNDWI, BSI, NDBI, NBR2, BAI, ...
    bands
        Mapping with the logical band names required by the index, e.g.
        ``{"nir": ..., "red": ...}``. Keys are the lower-case names in
        :data:`BAND_ALIASES` (``red``, ``green``, ``blue``, ``nir``,
        ``rededge``, ``swir``, ``swir2``).
    """
    key = name.upper()
    entry = _INDEX_DISPATCH.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown index: {name!r}. Supported: {', '.join(sorted(_INDEX_DISPATCH))}."
        )
    func, band_names = entry
    return func(*[bands[b] for b in band_names])


def required_bands(indices: list[str]) -> list[str]:
    """Return the sorted set of Sentinel-2 asset keys needed for given indices."""
    needed: set[str] = set()
    for name in indices:
        key = name.upper()
        entry = _INDEX_DISPATCH.get(key)
        if entry is None:
            raise ValueError(f"Unknown index: {name!r}.")
        needed.update(entry[1])
    return sorted({BAND_ALIASES[b] for b in needed})


def list_indices() -> list[str]:
    """Return all dispatchable index names (upper-case), in catalogue order."""
    return list(_INDEX_DISPATCH)
