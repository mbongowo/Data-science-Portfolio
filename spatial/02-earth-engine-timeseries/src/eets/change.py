"""Pure-numpy bitemporal change detection and burn-severity classification.

Given two cloud-robust composites — a *baseline* period and a *recent* period —
this module computes the per-pixel change in a spectral index, classifies each
pixel as loss / stable / gain against thresholds, and converts the pixel counts
to hectares. A parallel dNBR path classifies fire burn severity against the
standard USGS thresholds.

Sign convention
---------------
* :func:`change_map` returns ``after - before``. For NDVI a **negative** delta is
  vegetation *loss* (e.g. clearing), a **positive** delta is *gain* (regrowth).
* :func:`dnbr` returns ``pre - post`` so a **positive** dNBR is burn severity, in
  line with the USGS / FIREMON convention.

NaN handling: a pixel that is NaN in either composite is NaN in the change map
and is classified as ``0`` (stable / nodata) so it contributes no spurious
hectares. Everything here depends on numpy and the standard library only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# USGS / FIREMON dNBR burn-severity break points (Key & Benson, 2006). A dNBR
# below the low bound is treated as unburned (regrowth below it is ignored here).
# Classes returned by classify_burn_severity:
#   0 unburned, 1 low, 2 moderate-low, 3 moderate-high, 4 high.
DNBR_LOW = 0.10  # unburned / low boundary
DNBR_MOD_LOW = 0.27  # low / moderate-low boundary
DNBR_MOD_HIGH = 0.44  # moderate-low / moderate-high boundary
DNBR_HIGH = 0.66  # moderate-high / high boundary

_SEVERITY_NAMES = {
    0: "unburned",
    1: "low",
    2: "moderate_low",
    3: "moderate_high",
    4: "high",
}


def _check_2d_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate that ``a`` and ``b`` are 2-D and share a shape; return as float."""
    fa = np.asarray(a, dtype=np.float64)
    fb = np.asarray(b, dtype=np.float64)
    if fa.ndim != 2 or fb.ndim != 2:
        raise ValueError(f"inputs must be 2-D, got {fa.ndim}-D and {fb.ndim}-D")
    if fa.shape != fb.shape:
        raise ValueError(f"shape mismatch: {fa.shape} vs {fb.shape}")
    return fa, fb


def change_map(before_composite: np.ndarray, after_composite: np.ndarray) -> np.ndarray:
    """Per-pixel index change, ``after - before`` (NaN-propagating).

    A pixel that is NaN in either composite stays NaN, so missing data never
    looks like change.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.
    """
    before, after = _check_2d_pair(before_composite, after_composite)
    return after - before


def classify_change(
    delta: np.ndarray, loss_thresh: float, gain_thresh: float
) -> np.ndarray:
    """Classify a change map into loss / stable / gain.

    Parameters
    ----------
    delta:
        Per-pixel change (``after - before``) from :func:`change_map`.
    loss_thresh:
        Pixels with ``delta <= loss_thresh`` are *loss* (``-1``). Typically a
        negative number for NDVI (e.g. ``-0.2``).
    gain_thresh:
        Pixels with ``delta >= gain_thresh`` are *gain* (``+1``). Typically a
        positive number (e.g. ``+0.2``).

    Returns
    -------
    numpy.ndarray
        ``int8`` array: ``-1`` loss, ``+1`` gain, ``0`` stable. ``NaN`` pixels
        (no data) are ``0``.

    Raises
    ------
    ValueError
        If ``delta`` is not 2-D or ``loss_thresh > gain_thresh``.
    """
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"delta must be 2-D, got {arr.ndim}-D")
    if loss_thresh > gain_thresh:
        raise ValueError(
            f"loss_thresh ({loss_thresh}) must not exceed gain_thresh ({gain_thresh})"
        )
    out = np.zeros(arr.shape, dtype=np.int8)
    valid = ~np.isnan(arr)
    out[valid & (arr <= loss_thresh)] = -1
    out[valid & (arr >= gain_thresh)] = 1
    return out


def change_stats(classified: np.ndarray, pixel_size_m: float) -> dict[str, Any]:
    """Summarise a classified change map into pixel counts and hectares.

    Parameters
    ----------
    classified:
        ``int`` array from :func:`classify_change` (``-1`` loss, ``0`` stable,
        ``+1`` gain).
    pixel_size_m:
        Pixel side length in metres (10 for Sentinel-2). One pixel is
        ``pixel_size_m**2`` square metres.

    Returns
    -------
    dict
        Keys: ``loss_pixels``, ``gain_pixels``, ``loss_hectares``,
        ``gain_hectares``, ``loss_fraction``, ``gain_fraction`` (fractions are of
        the total pixel count).

    Raises
    ------
    ValueError
        If ``classified`` is not 2-D or ``pixel_size_m`` is not positive.
    """
    arr = np.asarray(classified)
    if arr.ndim != 2:
        raise ValueError(f"classified must be 2-D, got {arr.ndim}-D")
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be > 0, got {pixel_size_m}")

    loss_pixels = int(np.count_nonzero(arr == -1))
    gain_pixels = int(np.count_nonzero(arr == 1))
    total = int(arr.size)
    px_area_m2 = float(pixel_size_m) ** 2
    return {
        "loss_pixels": loss_pixels,
        "gain_pixels": gain_pixels,
        "loss_hectares": loss_pixels * px_area_m2 / 10_000.0,
        "gain_hectares": gain_pixels * px_area_m2 / 10_000.0,
        "loss_fraction": loss_pixels / total if total else 0.0,
        "gain_fraction": gain_pixels / total if total else 0.0,
    }


def dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    """Differenced Normalized Burn Ratio, ``pre - post`` (positive = burn).

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.
    """
    pre, post = _check_2d_pair(nbr_pre, nbr_post)
    return pre - post


def classify_burn_severity(dnbr_map: np.ndarray) -> np.ndarray:
    """Classify a dNBR map into USGS burn-severity classes.

    Break points (Key & Benson 2006; see module constants):

    =====================  ====================  =====
    dNBR range             severity              class
    =====================  ====================  =====
    ``< 0.10``             unburned              ``0``
    ``[0.10, 0.27)``       low                   ``1``
    ``[0.27, 0.44)``       moderate-low          ``2``
    ``[0.44, 0.66)``       moderate-high         ``3``
    ``>= 0.66``            high                  ``4``
    =====================  ====================  =====

    Parameters
    ----------
    dnbr_map:
        Per-pixel dNBR from :func:`dnbr`.

    Returns
    -------
    numpy.ndarray
        ``int8`` class array (``0``..``4``). ``NaN`` pixels map to ``0``
        (unburned / nodata).

    Raises
    ------
    ValueError
        If ``dnbr_map`` is not 2-D.
    """
    arr = np.asarray(dnbr_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"dnbr_map must be 2-D, got {arr.ndim}-D")
    out = np.zeros(arr.shape, dtype=np.int8)
    valid = ~np.isnan(arr)
    out[valid & (arr >= DNBR_LOW)] = 1
    out[valid & (arr >= DNBR_MOD_LOW)] = 2
    out[valid & (arr >= DNBR_MOD_HIGH)] = 3
    out[valid & (arr >= DNBR_HIGH)] = 4
    return out


def severity_stats(classes: np.ndarray, pixel_size_m: float) -> dict[str, Any]:
    """Hectares per burn-severity class.

    Parameters
    ----------
    classes:
        ``int`` class array from :func:`classify_burn_severity` (``0``..``4``).
    pixel_size_m:
        Pixel side length in metres.

    Returns
    -------
    dict
        Keys are the class names (``unburned``, ``low``, ``moderate_low``,
        ``moderate_high``, ``high``); values are hectares in that class.

    Raises
    ------
    ValueError
        If ``classes`` is not 2-D or ``pixel_size_m`` is not positive.
    """
    arr = np.asarray(classes)
    if arr.ndim != 2:
        raise ValueError(f"classes must be 2-D, got {arr.ndim}-D")
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be > 0, got {pixel_size_m}")
    px_area_ha = float(pixel_size_m) ** 2 / 10_000.0
    return {
        name: int(np.count_nonzero(arr == cls)) * px_area_ha
        for cls, name in _SEVERITY_NAMES.items()
    }
