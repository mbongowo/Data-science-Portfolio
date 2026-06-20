"""Breakpoint / disturbance detection on a (de-seasonalised) time series.

The core ``detect_breakpoint`` is **pure numpy/scipy-free**: it runs a CUSUM
(cumulative-sum-of-deviations) scan to locate the single most significant level
shift, and reports its index, date (if a time axis is supplied) and magnitude
(post-break mean minus pre-break mean). Only numpy is required; the function
is fully unit-testable without the geospatial stack.

A ``ruptures``-backed multiple-changepoint path is available behind a guarded
import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Breakpoint", "detect_breakpoint", "detect_breakpoints_ruptures"]


@dataclass
class Breakpoint:
    """A single detected level shift.

    Attributes
    ----------
    index:
        Position in the series *after* which the break occurs (the break sits
        between ``index`` and ``index + 1``).
    magnitude:
        Signed change in mean: ``mean(after) - mean(before)``. Negative values
        indicate a drop (e.g. vegetation loss from fire/deforestation).
    score:
        Detection statistic (max absolute normalised CUSUM); larger is more
        significant.
    date:
        Calendar date of the break if a time axis was supplied, else ``None``.
    detected:
        Whether ``score`` cleared the threshold.
    """

    index: int
    magnitude: float
    score: float
    date: Any | None = None
    detected: bool = True


def _cusum(y: np.ndarray) -> np.ndarray:
    """Standardised cumulative sum of mean-deviations.

    Returns an array the same length as ``y`` whose largest absolute value
    locates the most likely single level shift. The sum is normalised by
    ``std * sqrt(n)`` so that the statistic stays O(1) for a stationary noisy
    series (typically < 1) but grows large (several units) in the presence of a
    genuine level shift, giving a stable, length-independent threshold.
    """
    n = y.size
    mean = np.nanmean(y)
    std = np.nanstd(y)
    if not np.isfinite(std) or std == 0:
        std = 1.0
    dev = np.nan_to_num(y - mean, nan=0.0)
    return np.cumsum(dev) / (std * np.sqrt(n))


def detect_breakpoint(
    series: np.ndarray,
    times: np.ndarray | None = None,
    min_segment: int = 3,
    threshold: float = 1.0,
) -> Breakpoint:
    """Locate the single most significant level shift via CUSUM.

    Parameters
    ----------
    series:
        1-D series, typically the *residual* of a harmonic decomposition (so
        that seasonality does not masquerade as a break). NaNs are tolerated.
    times:
        Optional time coordinate (e.g. ``numpy.datetime64`` array) used to
        attach a ``date`` to the result. Must match ``series`` length.
    min_segment:
        Minimum number of samples required on each side of a candidate break;
        prevents spurious detections at the very ends.
    threshold:
        Minimum CUSUM score for the break to be marked ``detected``.

    Returns
    -------
    Breakpoint
        The most significant candidate (always returned; check ``.detected``).
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if n < 2 * min_segment + 1:
        raise ValueError(
            f"series too short ({n}) for min_segment={min_segment}"
        )
    if times is not None:
        times = np.asarray(times).ravel()
        if times.size != n:
            raise ValueError("series and times must have the same length")

    cusum = _cusum(y)

    # Candidate break positions, respecting the minimum segment length.
    lo, hi = min_segment - 1, n - min_segment
    scores = np.abs(cusum[lo:hi])
    rel_idx = int(np.argmax(scores))
    idx = lo + rel_idx
    score = float(scores[rel_idx])

    before = y[: idx + 1]
    after = y[idx + 1 :]
    magnitude = float(np.nanmean(after) - np.nanmean(before))

    date = None
    if times is not None:
        # The break is located at the first sample after the shift.
        date = times[min(idx + 1, n - 1)]

    return Breakpoint(
        index=idx,
        magnitude=magnitude,
        score=score,
        date=date,
        detected=score >= threshold,
    )


def detect_breakpoints_ruptures(
    series: np.ndarray,
    times: np.ndarray | None = None,
    penalty: float = 3.0,
    model: str = "l2",
) -> list[Breakpoint]:
    """Multiple-changepoint detection via ``ruptures`` (guarded import).

    Uses PELT to find an unknown number of breaks. Returns one ``Breakpoint``
    per detected change, with magnitude = mean(next segment) - mean(prev
    segment). Raises a clear error if ``ruptures`` is unavailable.
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "detect_breakpoints_ruptures requires the 'ruptures' package. "
            "Install it or use detect_breakpoint (pure numpy)."
        ) from exc

    y = np.asarray(series, dtype=float).ravel()
    y = np.nan_to_num(y, nan=float(np.nanmean(y)))
    if times is not None:
        times = np.asarray(times).ravel()

    algo = rpt.Pelt(model=model).fit(y)
    # ruptures returns segment end indices, the last being len(y).
    bkps = algo.predict(pen=penalty)

    results: list[Breakpoint] = []
    prev = 0
    for end in bkps[:-1]:
        before = y[prev:end]
        after = y[end:]
        if before.size == 0 or after.size == 0:
            continue
        magnitude = float(after.mean() - before.mean())
        date = times[end] if times is not None and end < times.size else None
        results.append(
            Breakpoint(
                index=int(end - 1),
                magnitude=magnitude,
                score=abs(magnitude),
                date=date,
                detected=True,
            )
        )
        prev = end
    return results
