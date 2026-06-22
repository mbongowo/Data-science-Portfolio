"""Performance analytics for an equity curve.

Three headline numbers, each defined plainly so a reader knows exactly what it
measures:

* :func:`total_return` — end-to-end growth of the equity curve.
* :func:`sharpe` — annualised risk-adjusted return; ``0.0`` when returns have
  no dispersion (a flat or constant series), avoiding a divide-by-zero.
* :func:`max_drawdown` — the largest peak-to-trough decline, as a positive
  fraction of the running peak.

Pure numpy/pandas; pinned by known-answer tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity: pd.Series | np.ndarray | list[float]) -> float:
    """Total return of an equity curve: ``equity[-1] / equity[0] - 1``.

    Returns ``0.0`` for an empty or single-point curve.
    """
    e = np.asarray(equity, dtype=float).ravel()
    if e.size < 2 or e[0] == 0.0:
        return 0.0
    return float(e[-1] / e[0] - 1.0)


def sharpe(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: float = 252.0,
    *,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of a per-period return series.

    ``Sharpe = sqrt(periods_per_year) * mean(excess) / std(excess)`` using the
    population standard deviation (``ddof=0``). When the excess returns have
    zero standard deviation (a constant series, including all-zero), the ratio
    is undefined and ``0.0`` is returned rather than raising.

    Parameters
    ----------
    returns:
        Per-period (per-bar) simple returns.
    periods_per_year:
        Scaling factor to annualise. For daily bars ~252; for 1-minute crypto
        bars ``365 * 24 * 60 = 525600``.
    risk_free:
        Per-period risk-free rate to subtract before scaling. Default ``0``.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    excess = r - risk_free
    sd = float(excess.std(ddof=0))
    if sd == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def max_drawdown(equity: pd.Series | np.ndarray | list[float]) -> float:
    """Maximum drawdown of an equity curve, as a positive fraction.

    The drawdown at each point is ``1 - equity / running_peak``; the maximum
    drawdown is the largest such value over the whole curve. A monotonically
    rising curve has drawdown ``0.0``.

    Example
    -------
    ``max_drawdown([100, 120, 90, 150])`` is ``0.25``: the peak is 120, the
    trough after it is 90, and ``(120 - 90) / 120 = 0.25``.
    """
    e = np.asarray(equity, dtype=float).ravel()
    if e.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(e)
    drawdown = 1.0 - e / running_peak
    return float(np.max(drawdown))
