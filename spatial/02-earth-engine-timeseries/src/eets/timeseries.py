"""Pure-numpy time-series and temporal-compositing core.

This is the part of the project that turns a *stack* of single-date index images
(dims ``(time, row, col)``) into the inputs the change detector needs:

* :func:`index_timeseries` — collapse each time step to one spatial-mean index
  value, NaN-aware, giving a 1-D series over time that shows the dip when
  vegetation is cleared.
* :func:`temporal_composite` — collapse the time axis per pixel to a single
  cloud-robust composite (median or mean over a period), NaN-aware, so scattered
  cloudy observations do not contaminate the baseline / recent images.
* :func:`mask_invalid` — set pixels flagged by the Sentinel-2 scene
  classification layer (SCL) — clouds, shadows, snow — to ``NaN`` before any
  index or composite is computed.

NaN is the missing-data sentinel throughout: masked / cloudy / nodata pixels are
``NaN`` and are skipped by the reducers, never read as a real 0. Everything here
depends on numpy and the standard library only.
"""

from __future__ import annotations

import numpy as np


def index_timeseries(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Per-time-step spatial mean of an index stack, ignoring NaN.

    Parameters
    ----------
    stack:
        An index array with a time axis, typically ``(time, row, col)``. Cloudy
        / masked pixels are ``NaN``.
    axis:
        The time axis (default ``0``). The spatial mean is taken over all *other*
        axes, leaving one value per time step.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``stack.shape[axis]``: the NaN-aware spatial mean
        index at each time step. A time step that is entirely NaN yields ``NaN``.

    Raises
    ------
    ValueError
        If ``stack`` has fewer than 2 dimensions (no spatial extent to average).
    """
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"stack must be >= 2-D (time + space), got {arr.ndim}-D")
    axis = axis % arr.ndim
    other = tuple(ax for ax in range(arr.ndim) if ax != axis)
    # all-NaN slices intentionally return NaN; suppress the numpy warning.
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=other)


def temporal_composite(
    stack: np.ndarray, agg: str = "median", axis: int = 0
) -> np.ndarray:
    """Reduce the time axis per pixel to a single cloud-robust composite.

    Parameters
    ----------
    stack:
        An index or band array with a time axis, typically ``(time, row, col)``.
        Cloudy / masked observations are ``NaN`` and are skipped.
    agg:
        ``"median"`` (default; robust to residual cloud) or ``"mean"``.
    axis:
        The time axis to collapse (default ``0``).

    Returns
    -------
    numpy.ndarray
        Array with the time axis removed: the per-pixel temporal median or mean.
        A pixel that is NaN at every time step stays ``NaN``.

    Raises
    ------
    ValueError
        If ``stack`` has fewer than 2 dimensions or ``agg`` is not recognised.
    """
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"stack must be >= 2-D (time + space), got {arr.ndim}-D")
    if agg not in ("median", "mean"):
        raise ValueError(f"agg must be 'median' or 'mean', got {agg!r}")
    reducer = np.nanmedian if agg == "median" else np.nanmean
    with np.errstate(invalid="ignore"):
        return reducer(arr, axis=axis)


def mask_invalid(
    band: np.ndarray,
    scl: np.ndarray,
    invalid_classes: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Set pixels whose SCL class is invalid to ``NaN``.

    The Sentinel-2 L2A scene-classification layer (SCL) labels each pixel with an
    integer class. Common classes to drop are 0 (no data), 1 (saturated), 3
    (cloud shadow), 8 (cloud medium probability), 9 (cloud high probability),
    10 (thin cirrus), and 11 (snow / ice).

    Parameters
    ----------
    band:
        A reflectance or index array.
    scl:
        The SCL class array, same shape as ``band``.
    invalid_classes:
        Iterable of integer SCL classes to mask out.

    Returns
    -------
    numpy.ndarray
        A float copy of ``band`` with invalid-class pixels set to ``NaN``. The
        input is not modified.

    Raises
    ------
    ValueError
        If ``band`` and ``scl`` shapes differ.
    """
    arr = np.asarray(band, dtype=np.float64)
    scl_arr = np.asarray(scl)
    if arr.shape != scl_arr.shape:
        raise ValueError(f"shape mismatch: band {arr.shape} vs scl {scl_arr.shape}")
    out = arr.copy()
    invalid = np.isin(scl_arr, np.asarray(list(invalid_classes)))
    out[invalid] = np.nan
    return out
