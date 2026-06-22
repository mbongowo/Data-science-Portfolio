"""Pure-numpy Otsu automatic thresholding.

Flood mapping with SAR comes down to one decision per pixel: water or not. On
Sentinel-1 backscatter (in decibels) a flooded surface is *dark* — smooth water
reflects radar away from the sensor — so a histogram of backscatter over a partly
flooded scene is **bimodal**: a low-backscatter water peak and a higher-backscatter
land peak. Otsu's method finds the threshold between those two peaks
automatically, with no hand-tuned cut-off, by maximising the between-class
variance of the two groups the threshold defines.

Algorithm (Otsu 1979)
----------------------
1. Build a histogram of the finite (non-NaN) values into ``bins`` bins.
2. Normalise it to a probability mass function ``p`` over bin centres.
3. For every possible split ``t`` between bins, the pixels below the split form
   "class 0" and the rest "class 1". Let ``w0(t)``/``w1(t)`` be the class weights
   (summed probabilities) and ``mu0(t)``/``mu1(t)`` their mean bin centres. The
   between-class variance is ``w0 * w1 * (mu0 - mu1) ** 2``.
4. The Otsu threshold is the bin-centre split that **maximises** that variance —
   equivalently it minimises the within-class variance. It lands in the valley
   between the two modes.

Everything here depends on numpy and the standard library only. NaN values are
ignored (treated as missing), matching the rest of the core.
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Return the Otsu between-class-variance-maximising threshold.

    Builds a ``bins``-bin histogram of the finite values and returns the bin
    centre that best separates a low-value class from a high-value class. For
    SAR backscatter in dB the returned value is the water/land cut-off: pixels
    **below** it are water (see :func:`floodmap.water.water_mask`).

    Parameters
    ----------
    values:
        Array of any shape (e.g. a backscatter-in-dB image). ``NaN`` entries are
        ignored. Must contain at least two distinct finite values.
    bins:
        Number of histogram bins (default 256). More bins give a finer threshold
        at the cost of noisier histograms on small samples.

    Returns
    -------
    float
        The Otsu threshold, a value between the data minimum and maximum.

    Raises
    ------
    ValueError
        If ``bins < 2`` or there are fewer than two distinct finite values.
    """
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("no finite values to threshold")
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        raise ValueError("need at least two distinct finite values to threshold")

    counts, edges = np.histogram(finite, bins=bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = counts.astype(np.float64)
    p /= p.sum()

    # Cumulative class-0 weight and cumulative (probability * center) up to each
    # split. Class 0 is bins [0..t], class 1 is bins [t+1..end].
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    cum_mean = np.cumsum(p * centers)
    total_mean = cum_mean[-1]

    # Valid splits have non-empty classes on both sides.
    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.zeros_like(w0)
    mu1 = np.zeros_like(w1)
    np.divide(cum_mean, w0, out=mu0, where=valid)
    np.divide(total_mean - cum_mean, w1, out=mu1, where=valid)

    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~valid] = -np.inf
    best = int(np.argmax(between))
    return float(centers[best])


def histogram_modes(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Return the two dominant histogram mode locations (low, high).

    A small diagnostic helper: split the histogram at the Otsu threshold and
    report the most-populated bin centre on each side. For a clean bimodal
    backscatter scene these approximate the water peak and the land peak, and the
    Otsu threshold sits between them.

    Parameters
    ----------
    values:
        Array of any shape; ``NaN`` ignored.
    bins:
        Number of histogram bins.

    Returns
    -------
    tuple of float
        ``(low_mode, high_mode)`` bin-centre estimates, ``low_mode <= high_mode``.

    Raises
    ------
    ValueError
        If there are fewer than two distinct finite values.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("no finite values")
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        raise ValueError("need at least two distinct finite values")

    counts, edges = np.histogram(finite, bins=bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    thresh = otsu_threshold(finite, bins=bins)

    low_side = centers <= thresh
    high_side = ~low_side
    low_counts = np.where(low_side, counts, -1)
    high_counts = np.where(high_side, counts, -1)
    low_mode = float(centers[int(np.argmax(low_counts))])
    high_mode = float(centers[int(np.argmax(high_counts))])
    return low_mode, high_mode
