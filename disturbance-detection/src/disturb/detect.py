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

__all__ = [
    "Breakpoint",
    "detect_breakpoint",
    "detect_breakpoints_binseg",
    "recovery_time",
    "detect_breakpoints_ruptures",
]


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


def detect_breakpoints_binseg(
    series: np.ndarray,
    times: np.ndarray | None = None,
    max_breaks: int = 3,
    threshold: float = 1.0,
    min_segment: int = 3,
) -> list[Breakpoint]:
    """Find multiple changepoints by recursive binary segmentation.

    Pure-numpy multiple-changepoint detection built entirely on the existing
    single-break :func:`detect_breakpoint` CUSUM scan - no ``ruptures``
    dependency. The algorithm:

    1. Run :func:`detect_breakpoint` on the whole series.
    2. If the break clears ``threshold``, accept it and recurse independently
       into the left segment (``[start, idx]``) and the right segment
       (``[idx + 1, end]``).
    3. Stop when a segment is too short to hold another break, when no break
       clears the threshold, or when ``max_breaks`` accepted breaks is reached.

    Parameters
    ----------
    series:
        1-D series (typically a harmonic residual). NaNs are tolerated by the
        underlying CUSUM scan.
    times:
        Optional time axis, used only to attach calendar ``date`` values.
    max_breaks:
        Maximum number of breakpoints to return.
    threshold:
        Minimum CUSUM score for a candidate to be accepted, passed through to
        :func:`detect_breakpoint`.
    min_segment:
        Minimum samples on each side of a candidate break.

    Returns
    -------
    list[Breakpoint]
        Accepted breakpoints sorted by ascending index. The ``magnitude`` of
        each is recomputed as ``mean(next segment) - mean(prev segment)`` using
        the *local* segment boundaries, so adjacent steps do not contaminate
        one another. Empty if nothing clears the threshold.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if times is not None:
        times = np.asarray(times).ravel()
        if times.size != n:
            raise ValueError("series and times must have the same length")

    min_len = 2 * min_segment + 1
    accepted: list[int] = []

    # Work queue of half-open segments [start, end) to scan.
    stack: list[tuple[int, int]] = [(0, n)]
    while stack and len(accepted) < max_breaks:
        start, end = stack.pop()
        if end - start < min_len:
            continue
        bp = detect_breakpoint(
            y[start:end],
            times=None,
            min_segment=min_segment,
            threshold=threshold,
        )
        if not bp.detected:
            continue
        idx = start + bp.index  # break sits after global index `idx`
        accepted.append(idx)
        # Recurse into the two child segments.
        stack.append((start, idx + 1))
        stack.append((idx + 1, end))

    accepted = sorted(accepted)[:max_breaks]

    # Recompute magnitude on the final local segments defined by the breaks.
    bounds = [0, *[(b + 1) for b in accepted], n]
    results: list[Breakpoint] = []
    for k, idx in enumerate(accepted):
        before = y[bounds[k] : idx + 1]
        after = y[idx + 1 : bounds[k + 2]]
        magnitude = float(np.nanmean(after) - np.nanmean(before))
        date = times[min(idx + 1, n - 1)] if times is not None else None
        results.append(
            Breakpoint(
                index=int(idx),
                magnitude=magnitude,
                score=abs(magnitude),
                date=date,
                detected=True,
            )
        )
    return results


def recovery_time(
    series: np.ndarray,
    break_index: int,
    tolerance: float | None = None,
) -> int | None:
    """Samples until the series returns to its pre-break level.

    After a disturbance (a drop), vegetation may recover. This counts how many
    samples *after* the break the series first comes back to within
    ``tolerance`` of the pre-break mean level and stays defined.

    Parameters
    ----------
    series:
        1-D series (NDVI-like). NaNs after the break are skipped.
    break_index:
        Index *after which* the break occurred (the break sits between
        ``break_index`` and ``break_index + 1``), matching
        :attr:`Breakpoint.index`.
    tolerance:
        Absolute band around the pre-break mean that counts as "recovered".
        Defaults to one standard deviation of the pre-break segment (or a small
        floor if that segment is flat).

    Returns
    -------
    int or None
        Number of samples after the break at which the post-break value first
        re-enters ``[pre_mean - tol, pre_mean + tol]`` (1-based: 1 means the
        very next sample). ``None`` if it never recovers within the series, or
        if there is no post-break data.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if break_index < 0 or break_index >= n - 1:
        return None

    before = y[: break_index + 1]
    pre_mean = float(np.nanmean(before))
    if tolerance is None:
        pre_std = float(np.nanstd(before))
        tolerance = pre_std if np.isfinite(pre_std) and pre_std > 0 else 1e-9

    after = y[break_index + 1 :]
    for offset, value in enumerate(after, start=1):
        if not np.isfinite(value):
            continue
        if abs(value - pre_mean) <= tolerance:
            return offset
    return None


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
