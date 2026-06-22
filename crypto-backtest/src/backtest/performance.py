"""Performance analytics for an equity curve.

Each number is defined plainly so a reader knows exactly what it measures:

* :func:`total_return` — end-to-end growth of the equity curve.
* :func:`sharpe` — annualised risk-adjusted return; ``0.0`` when returns have
  no dispersion (a flat or constant series), avoiding a divide-by-zero.
* :func:`max_drawdown` — the largest peak-to-trough decline, as a positive
  fraction of the running peak.
* :func:`sortino` — like Sharpe but penalising only downside deviation;
  ``+inf`` when there is no downside and the mean is positive.
* :func:`calmar` — annualised (CAGR) return over max drawdown.
* :func:`win_rate` — fraction of strictly-positive periods.
* :func:`turnover` — average per-bar position change (what costs are charged on).
* :func:`exposure` — fraction of bars holding a non-zero position.

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


def sortino(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: float = 252.0,
    *,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sortino ratio: excess return over *downside* deviation only.

    ``Sortino = sqrt(periods_per_year) * mean(excess) / downside_std`` where the
    downside deviation is the population RMS of the **negative** excess returns
    (positive excess returns contribute ``0`` to the denominator, not removed
    from the count). This is the convention that makes the metric hand-derivable.

    Edge cases (the known-answers the tests pin):

    * No downside at all (every excess return ``>= 0``) and a positive mean ->
      the ratio is ``+inf`` (``float('inf')``).
    * A flat/zero series (zero mean and zero downside) -> ``0.0``, not ``nan``.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    excess = r - risk_free
    mean = float(excess.mean())
    downside = np.minimum(excess, 0.0)
    downside_var = float(np.mean(downside**2))
    downside_std = float(np.sqrt(downside_var))
    if downside_std == 0.0:
        # No downside risk. Positive mean -> unbounded ratio; zero mean -> 0.
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return float(np.sqrt(periods_per_year) * mean / downside_std)


def calmar(
    equity: pd.Series | np.ndarray | list[float],
    periods_per_year: float = 252.0,
) -> float:
    """Calmar ratio: annualised (CAGR) return divided by max drawdown.

    The numerator is the compound annual growth rate implied by the equity
    curve over its length; the denominator is :func:`max_drawdown` (a positive
    fraction). A curve with no drawdown (monotonically rising) has an undefined
    ratio, returned as ``+inf`` when it grew and ``0.0`` when it was flat.

    Returns ``0.0`` for an empty or single-point curve.
    """
    e = np.asarray(equity, dtype=float).ravel()
    if e.size < 2 or e[0] == 0.0:
        return 0.0
    n_periods = e.size - 1
    growth = e[-1] / e[0]
    cagr = float(growth ** (periods_per_year / n_periods) - 1.0)
    mdd = max_drawdown(e)
    if mdd == 0.0:
        if cagr > 0.0:
            return float("inf")
        if cagr < 0.0:
            return float("-inf")
        return 0.0
    return float(cagr / mdd)


def win_rate(returns: pd.Series | np.ndarray | list[float]) -> float:
    """Fraction of periods with a strictly positive return, in ``[0, 1]``.

    Zeros and losses do not count as wins. Returns ``0.0`` for an empty series.
    Example: ``win_rate([0.1, -0.2, 0.0, 0.3])`` is ``2 / 4 = 0.5``.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    return float(np.count_nonzero(r > 0.0) / r.size)


def turnover(positions: pd.Series | np.ndarray | list[float]) -> float:
    """Average per-bar turnover of a position series.

    Turnover at bar ``t`` is ``|position[t] - position[t-1]|``, measured against
    a flat (``0``) starting position on the first bar — exactly the quantity the
    engine charges costs on. The returned figure is the mean of those per-bar
    changes, so a strategy that enters once and holds has a tiny average and one
    that flips every bar has a large one.

    Returns ``0.0`` for an empty series.
    Example: positions ``[0, 1, 1, 0]`` -> changes ``[0, 1, 0, 1]`` -> mean
    ``0.5``.
    """
    p = np.asarray(positions, dtype=float).ravel()
    if p.size == 0:
        return 0.0
    changes = np.abs(np.diff(p, prepend=0.0))
    return float(changes.mean())


def exposure(positions: pd.Series | np.ndarray | list[float]) -> float:
    """Fraction of bars with a non-zero position (market exposure), in ``[0, 1]``.

    Counts any bar whose position is not exactly ``0`` (long or short alike).
    Returns ``0.0`` for an empty series.
    Example: positions ``[0, 1, 1, 0]`` -> ``2 / 4 = 0.5``.
    """
    p = np.asarray(positions, dtype=float).ravel()
    if p.size == 0:
        return 0.0
    return float(np.count_nonzero(p != 0.0) / p.size)
