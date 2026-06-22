"""Per-pixel anomaly (z-score) computation against a climatological baseline.

Given a stack of an index over a baseline window and a stack over the target
window, the anomaly is the standardised departure of each target observation
from the baseline mean::

    z = (value - baseline_mean) / baseline_std

The math is pure numpy/xarray so it is hand-verifiable in the unit tests.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

ArrayT = TypeVar("ArrayT")


def baseline_statistics(
    baseline: ArrayT,
    dim: str = "time",
) -> tuple[ArrayT, ArrayT]:
    """Return (mean, std) of the baseline stack reduced over ``dim``.

    For numpy input, ``dim`` is interpreted as axis 0 and NaNs are ignored
    (population std, ddof=0). For xarray input the named dimension is reduced
    with ``skipna=True`` and the computation stays lazy (no ``.compute()``).
    """
    if isinstance(baseline, np.ndarray):
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(baseline, axis=0)
            std = np.nanstd(baseline, axis=0)
        return mean, std  # type: ignore[return-value]
    mean = baseline.mean(dim=dim, skipna=True)
    std = baseline.std(dim=dim, skipna=True)
    return mean, std  # type: ignore[return-value]


def zscore_anomaly(
    value: ArrayT,
    baseline_mean: ArrayT,
    baseline_std: ArrayT,
) -> ArrayT:
    """Standardised anomaly ``(value - baseline_mean) / baseline_std``.

    Where ``baseline_std == 0`` the result is NaN (the pixel carries no
    variability information and a z-score is undefined).
    """
    if (
        isinstance(value, np.ndarray)
        or isinstance(baseline_mean, np.ndarray)
        or isinstance(baseline_std, np.ndarray)
    ):
        v = np.asarray(value, dtype="float64")
        m = np.asarray(baseline_mean, dtype="float64")
        s = np.asarray(baseline_std, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(s == 0, np.nan, (v - m) / s)
        return out  # type: ignore[return-value]
    # xarray path: NaN propagates where std == 0 via .where().
    z = (value - baseline_mean) / baseline_std  # type: ignore[operator]
    return z.where(baseline_std != 0)  # type: ignore[attr-defined]


# Scale factor making the MAD a consistent estimator of the standard deviation
# for normally distributed data: 1 / Phi^-1(0.75) = 1.482602...
_MAD_SCALE = 1.4826


def robust_zscore(
    value: ArrayT,
    baseline: ArrayT,
    dim: str = "time",
) -> ArrayT:
    """Outlier-resistant z-score using the baseline median and MAD.

    The baseline is reduced over ``dim`` (axis 0 for numpy) to a per-pixel
    median and a scaled median absolute deviation,
    ``MAD = 1.4826 * median(|x - median(x)|)``, then::

        z = (value - median) / MAD

    The 1.4826 factor makes the MAD match the standard deviation for normal
    data, so this z is comparable to :func:`zscore_anomaly` but unmoved by a few
    extreme baseline observations. Where ``MAD == 0`` (no robust spread) the
    result is NaN. NaNs in the baseline are ignored.

    Worked example: baseline ``[1, 2, 4, 100]`` -> median 3,
    ``|x - 3| = [2, 1, 1, 97]`` -> median 1.5, MAD ``= 1.4826 * 1.5 = 2.2239``.
    A value of 3 gives z 0; a value of ``3 + 2.2239`` gives z 1.
    """
    if isinstance(baseline, np.ndarray) or isinstance(value, np.ndarray):
        b = np.asarray(baseline, dtype="float64")
        v = np.asarray(value, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            med = np.nanmedian(b, axis=0)
            mad = _MAD_SCALE * np.nanmedian(np.abs(b - med), axis=0)
            out = np.where(mad == 0, np.nan, (v - med) / mad)
        return out  # type: ignore[return-value]
    # xarray path.
    med = baseline.median(dim=dim, skipna=True)
    mad = _MAD_SCALE * abs(baseline - med).median(dim=dim, skipna=True)
    z = (value - med) / mad  # type: ignore[operator]
    return z.where(mad != 0)  # type: ignore[attr-defined]


def anomaly_fraction(z: ArrayT, threshold: float = 2.0) -> float:
    """Fraction of finite pixels whose anomaly exceeds ``|z| > threshold``.

    NaN pixels (masked / undefined z) are excluded from both numerator and
    denominator. Returns 0.0 when there are no finite pixels.
    """
    arr = np.asarray(z, dtype="float64")
    finite = np.isfinite(arr)
    n = int(finite.sum())
    if n == 0:
        return 0.0
    flagged = int((np.abs(arr[finite]) > threshold).sum())
    return flagged / n


def classify_anomaly(z: ArrayT, threshold: float = 2.0) -> ArrayT:
    """Classify a z-score field into -1 (loss), 0 (none), +1 (gain).

    ``z < -threshold`` -> -1, ``z > +threshold`` -> +1, otherwise 0. NaN
    pixels (undefined z) classify as 0 (not flagged). Returns an int8 array.
    """
    arr = np.asarray(z, dtype="float64")
    out = np.zeros(arr.shape, dtype="int8")
    with np.errstate(invalid="ignore"):
        out = np.where(arr > threshold, 1, out)
        out = np.where(arr < -threshold, -1, out)
    return out.astype("int8")  # type: ignore[return-value]


def anomaly_cube(
    target: ArrayT,
    baseline: ArrayT,
    dim: str = "time",
) -> ArrayT:
    """Compute the z-score anomaly of every ``target`` time slice.

    Parameters
    ----------
    target
        Stack of the index over the target window (dim ``time`` first / named).
    baseline
        Stack of the index over the baseline window, same spatial grid.
    dim
        Time dimension name (xarray) or assumed axis 0 (numpy).
    """
    mean, std = baseline_statistics(baseline, dim=dim)
    return zscore_anomaly(target, mean, std)
