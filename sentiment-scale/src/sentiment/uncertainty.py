r"""Uncertainty quantification for aggregated sentiment (pure numpy).

A weekly mean sentiment score is a point estimate; quoting it without an error
bar invites over-reading noise. This module provides a non-parametric
**bootstrap** confidence interval for the mean of a sample of scores, so a
reported level can carry an honest ``[lo, hi]`` band.

The bootstrap resamples the observed scores *with replacement* ``n_boot`` times,
takes the mean of each resample, and reads the ``alpha/2`` and ``1 - alpha/2``
empirical percentiles of those bootstrap means. For a 95% interval, pass
``alpha = 0.05``. The procedure is seeded, so the interval is reproducible.

This is the percentile bootstrap: it makes no normality assumption and is exactly
the tool to attach to a mean-of-scores aggregate.
"""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    scores: np.ndarray | list[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    r"""Percentile bootstrap confidence interval for the mean of ``scores``.

    Parameters
    ----------
    scores:
        A 1-D array (or list) of per-document sentiment scores.
    n_boot:
        Number of bootstrap resamples. Larger is tighter/steadier; 1000 is a
        sensible default.
    seed:
        Seed for ``numpy.random.default_rng``; makes the interval reproducible.
    alpha:
        Significance level. The interval is the ``[alpha/2, 1 - alpha/2]``
        percentile band of the bootstrap means, i.e. a ``100 * (1 - alpha)%``
        confidence interval. ``alpha = 0.05`` gives a 95% interval.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        The lower and upper confidence bounds for the mean, with ``lo <= hi``.

    Raises
    ------
    ValueError
        If ``scores`` is empty, ``n_boot < 1``, or ``alpha`` is not in ``(0, 1)``.

    Notes
    -----
    With ``n`` observations each resample draws ``n`` of them with replacement.
    A degenerate sample (all values equal) returns that value for both bounds.

    Examples
    --------
    >>> lo, hi = bootstrap_mean_ci([1.0, 1.0, 1.0, 1.0], n_boot=200, seed=0)
    >>> (lo, hi)
    (1.0, 1.0)
    """
    arr = np.asarray(scores, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("bootstrap_mean_ci requires at least one score.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = arr.size
    # (n_boot, n) matrix of resample indices, vectorised for speed.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)

    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi
