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
