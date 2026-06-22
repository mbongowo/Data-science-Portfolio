"""Windowing and funnel primitives for event streams (pure Python).

This module is the interpretation-critical core of the pipeline. It has no
third-party dependency beyond the standard library, so it is always importable
and is the basis of the known-answer unit tests. The functions here operate on
plain Python lists and dicts of already-decoded events; the streaming engine
(Kafka + Spark) that feeds them in production lives in :mod:`clickstream.pipeline`
and is imported lazily there, never here.

The four primitives mirror the standard streaming aggregations:

* :func:`tumbling_counts` fixed, non-overlapping windows.
* :func:`sliding_counts` overlapping windows that advance by a slide.
* :func:`sessionize` groups events into sessions split by an inactivity gap.
* :func:`funnel` counts how many users reached each prefix of an ordered funnel.

Timestamps are epoch *seconds* throughout. Window boundaries are aligned to the
epoch: a window of width ``window_s`` starting at ``t`` covers ``[t, t +
window_s)``, half-open, so an event exactly on a boundary belongs to the later
window. That convention is what the tests check.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence


def tumbling_counts(
    events: Sequence[tuple[float, Hashable]], window_s: float
) -> dict[float, int]:
    """Count events per fixed, non-overlapping (tumbling) window.

    Each event is assigned to exactly one window whose start is
    ``floor(ts / window_s) * window_s``. The window covers ``[start, start +
    window_s)`` (half-open), so an event whose timestamp equals a boundary falls
    in the later window.

    Parameters
    ----------
    events:
        Sequence of ``(timestamp_seconds, key)`` pairs. The ``key`` is ignored
        for the count (it is carried so the same event type can feed other
        aggregations); only the timestamp decides the window.
    window_s:
        Window width in seconds. Must be positive.

    Returns
    -------
    dict
        Mapping ``{window_start: count}``. Windows with no events are absent.

    Raises
    ------
    ValueError
        If ``window_s`` is not positive.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive.")

    counts: dict[float, int] = defaultdict(int)
    for ts, _key in events:
        start = (ts // window_s) * window_s
        counts[start] += 1
    return dict(counts)


def sliding_counts(
    events: Sequence[tuple[float, Hashable]], window_s: float, slide_s: float
) -> dict[float, int]:
    """Count events per overlapping (sliding) window.

    Windows start at integer multiples of ``slide_s`` and are ``window_s`` wide.
    An event at time ``ts`` belongs to every window ``[start, start + window_s)``
    that contains it; with ``window_s > slide_s`` an event is therefore counted
    in several windows.

    The set of candidate window starts for an event are the multiples of
    ``slide_s`` in ``(ts - window_s, ts]``. Concretely the earliest window that
    still contains ``ts`` starts at ``ceil((ts - window_s + slide_s) / slide_s)
    * slide_s`` is avoided here in favour of a direct, readable enumeration.

    Parameters
    ----------
    events:
        Sequence of ``(timestamp_seconds, key)`` pairs.
    window_s:
        Window width in seconds. Must be positive.
    slide_s:
        Step between consecutive window starts in seconds. Must be positive and
        no greater than ``window_s`` (a larger slide would skip events).

    Returns
    -------
    dict
        Mapping ``{window_start: count}`` for every window start that contains
        at least one event.

    Raises
    ------
    ValueError
        If ``window_s`` or ``slide_s`` is not positive, or ``slide_s >
        window_s``.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive.")
    if slide_s <= 0:
        raise ValueError("slide_s must be positive.")
    if slide_s > window_s:
        raise ValueError("slide_s must not exceed window_s (events would be skipped).")

    counts: dict[float, int] = defaultdict(int)
    for ts, _key in events:
        # The first slide-aligned start at or before ts.
        first = (ts // slide_s) * slide_s
        start = first
        # Walk back while the window still covers ts: start <= ts < start+window_s.
        while start > ts - window_s:
            if start <= ts:
                counts[start] += 1
            start -= slide_s
    return dict(counts)


def sessionize(timestamps: Sequence[float], gap_s: float) -> list[int]:
    """Assign each event to a session, splitting on an inactivity gap.

    Walking the (sorted) timestamps in order, the session id increments whenever
    the gap to the previous event *exceeds* ``gap_s``. The first event is always
    session 0. A gap of exactly ``gap_s`` does not start a new session.

    Example
    -------
    ``sessionize([0, 1, 2, 10, 11], gap_s=5)`` returns ``[0, 0, 0, 1, 1]``: the
    jump from 2 to 10 is 8 > 5, so a new session starts at index 3.

    Parameters
    ----------
    timestamps:
        Non-decreasing sequence of epoch-second timestamps.
    gap_s:
        Inactivity threshold in seconds. Must be non-negative.

    Returns
    -------
    list of int
        Session ids, one per input timestamp, starting at 0 and increasing.

    Raises
    ------
    ValueError
        If ``gap_s`` is negative or the timestamps are not non-decreasing.
    """
    if gap_s < 0:
        raise ValueError("gap_s must be non-negative.")

    sessions: list[int] = []
    current = 0
    prev: float | None = None
    for ts in timestamps:
        if prev is not None:
            if ts < prev:
                raise ValueError("timestamps must be sorted non-decreasing.")
            if ts - prev > gap_s:
                current += 1
        sessions.append(current)
        prev = ts
    return sessions


def funnel(
    user_events: Mapping[Hashable, Sequence[str]], steps: Sequence[str]
) -> list[int]:
    """Count users reaching each prefix of an ordered funnel.

    A user "reaches" step ``k`` if the funnel's first ``k`` step names appear in
    that user's event list **in order** (not necessarily contiguously). The
    return is the count of users reaching at least each prefix length, from 1 up
    to ``len(steps)``. Because reaching step ``k`` implies reaching step
    ``k - 1``, the returned list is monotonically non-increasing.

    Parameters
    ----------
    user_events:
        Mapping of ``user -> ordered list of event names`` for that user.
    steps:
        The ordered funnel step names, e.g. ``["view", "add_to_cart",
        "purchase"]``.

    Returns
    -------
    list of int
        Length-``len(steps)`` list; element ``k`` is the number of users who
        reached at least the first ``k + 1`` steps.

    Raises
    ------
    ValueError
        If ``steps`` is empty.
    """
    if len(steps) == 0:
        raise ValueError("steps must be non-empty.")

    reached = [0] * len(steps)
    for events in user_events.values():
        depth = _funnel_depth(events, steps)
        # The user reached every prefix up to `depth`.
        for k in range(depth):
            reached[k] += 1
    return reached


def _funnel_depth(events: Sequence[str], steps: Sequence[str]) -> int:
    """Return how many leading funnel steps appear in order within ``events``."""
    depth = 0
    for ev in events:
        if depth < len(steps) and ev == steps[depth]:
            depth += 1
    return depth
