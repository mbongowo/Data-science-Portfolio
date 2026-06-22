"""Robust monotonic-trend statistics (pure numpy).

Two classical non-parametric tools, both useful for asking *is this NDVI
series trending, and how fast?* without assuming Gaussian residuals or
evenly-spaced samples:

* :func:`theil_sen_slope` - the **Theil-Sen estimator**, the median of the
  slopes ``(y_j - y_i) / (t_j - t_i)`` taken over all pairs ``i < j``. It is
  robust to outliers (up to ~29% of the points can be arbitrary) and, unlike
  ordinary least squares, is not dragged around by a single bad observation
  (an uncaught cloud, a saturated pixel).

* :func:`mann_kendall` - the **Mann-Kendall test** for the presence of a
  monotonic trend. It counts, over all pairs ``i < j``, whether ``y_j`` rises
  or falls relative to ``y_i``:

      S = sum_{i<j} sign(y_j - y_i)

  Under the null hypothesis of no trend, ``S`` has mean 0 and variance

      Var(S) = n (n - 1) (2n + 5) / 18

  (ignoring ties). The standardised statistic with the usual continuity
  correction is

      z = (S - 1) / sqrt(Var(S))   if S > 0
      z = (S + 1) / sqrt(Var(S))   if S < 0
      z = 0                        if S == 0

  and the two-sided p-value is ``2 * (1 - Phi(|z|))`` where ``Phi`` is the
  standard-normal CDF. A small p-value means a significant monotonic trend.

Both depend only on numpy, so they sit in the unit-tested pure core.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["theil_sen_slope", "mann_kendall", "MannKendallResult"]

from dataclasses import dataclass


def theil_sen_slope(y: np.ndarray, t: np.ndarray | None = None) -> float:
    """Theil-Sen robust slope: median of all pairwise slopes.

    Parameters
    ----------
    y : numpy.ndarray
        1-D series of observations. NaNs are dropped before the computation.
    t : numpy.ndarray, optional
        Time coordinate (same length as ``y``). Defaults to ``0..n-1``.

    Returns
    -------
    float
        The median of ``(y_j - y_i) / (t_j - t_i)`` over all pairs with
        ``t_j != t_i``. ``nan`` if fewer than two valid pairs remain.

    Notes
    -----
    For a perfect line ``y = a + b t`` every pairwise slope equals ``b``, so the
    estimator returns ``b`` exactly. The breakdown point is ~29%: contaminating
    fewer than that fraction of points cannot move the median slope to infinity.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if t is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.asarray(t, dtype=float).ravel()
    if t.size != n:
        raise ValueError("y and t must have the same length")

    finite = np.isfinite(y) & np.isfinite(t)
    y = y[finite]
    t = t[finite]
    if y.size < 2:
        return float("nan")

    # All pairs i < j as an upper-triangular set of index pairs.
    i, j = np.triu_indices(y.size, k=1)
    dt = t[j] - t[i]
    good = dt != 0.0
    if not np.any(good):
        return float("nan")
    slopes = (y[j] - y[i])[good] / dt[good]
    return float(np.median(slopes))


@dataclass
class MannKendallResult:
    """Outcome of a Mann-Kendall monotonic-trend test.

    Attributes
    ----------
    trend:
        ``"increasing"``, ``"decreasing"`` or ``"no trend"`` at the 5% level.
    s:
        The Mann-Kendall S statistic (signed pair count).
    p_value:
        Two-sided normal-approximation p-value.
    z:
        Standardised statistic with continuity correction.
    """

    trend: str
    s: int
    p_value: float
    z: float


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (stdlib ``math.erf``)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_kendall(y: np.ndarray, alpha: float = 0.05) -> MannKendallResult:
    """Mann-Kendall test for a monotonic trend (normal approximation).

    Parameters
    ----------
    y : numpy.ndarray
        1-D series. NaNs are dropped first.
    alpha : float, optional
        Significance level for the reported ``trend`` label (default 0.05).

    Returns
    -------
    MannKendallResult
        Trend label, S statistic, p-value and standardised z.

    Notes
    -----
    Uses the tie-corrected variance

        Var(S) = [n(n-1)(2n+5) - sum_t g_t (g_t-1)(2 g_t+5)] / 18

    where ``g_t`` is the size of the ``t``-th group of tied values, and the
    continuity-corrected z given in the module docstring. For a strictly
    increasing series S is maximal and positive; for a flat series S = 0,
    z = 0 and p = 1.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        # Too short to say anything; S is still defined but the normal
        # approximation is meaningless.
        return MannKendallResult(trend="no trend", s=0, p_value=1.0, z=0.0)

    i, j = np.triu_indices(n, k=1)
    s = int(np.sum(np.sign(y[j] - y[i])))

    # Tie correction.
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1.0) * (2.0 * counts + 5.0)))
    var_s = (n * (n - 1.0) * (2.0 * n + 5.0) - tie_term) / 18.0

    if var_s <= 0.0:
        return MannKendallResult(trend="no trend", s=s, p_value=1.0, z=0.0)

    if s > 0:
        z = (s - 1.0) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / math.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    if p_value < alpha and s > 0:
        trend = "increasing"
    elif p_value < alpha and s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return MannKendallResult(trend=trend, s=s, p_value=float(p_value), z=float(z))
