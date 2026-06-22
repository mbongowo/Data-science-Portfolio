"""Bounded-memory and event-time analytics primitives (pure Python).

This module deepens the interpretation-critical core with four capabilities a
reviewer expects from a real streaming-analytics layer, all in plain Python with
no third-party dependency (so they are always importable and unit-tested with
hand-derived values):

* :func:`top_k_heavy_hitters` finds the most frequent keys in one pass using the
  bounded-counter Misra-Gries summary, then returns exact counts.
* :func:`reorder_within_lateness` re-orders an out-of-order event stream into
  timestamp order within an allowed-lateness bound and reports events that were
  too late to place.
* :func:`funnel_time_to_convert` measures the median time between consecutive
  completed funnel steps.
* :func:`retention` measures the fraction of users active in consecutive periods.

Timestamps are epoch *seconds* throughout, matching :mod:`clickstream.windows`
and :mod:`clickstream.watermark`. The Kafka + Spark engine in
:mod:`clickstream.pipeline` would compute equivalents at scale; this module is
the reference an analyst can read and reason about.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence


def top_k_heavy_hitters(
    keys: Sequence[Hashable], k: int, counters: int | None = None
) -> list[tuple[Hashable, int]]:
    """Find the ``k`` most frequent keys via the Misra-Gries summary.

    Misra-Gries keeps at most ``counters`` active tallies in a single pass, using
    memory bounded by ``counters`` regardless of how many *distinct* keys the
    stream contains. For each key: if it is already tracked, increment it; else if
    a counter is free, start it at one; else decrement *every* counter and drop
    those that hit zero. After the pass the surviving keys are guaranteed to
    include every key occurring strictly more than ``len(keys) / (counters + 1)``
    times. A second pass then counts the *exact* frequency of the surviving
    candidates and the top ``k`` by ``(-count, key)`` are returned, so the counts
    reported are true frequencies, not Misra-Gries lower bounds.

    Honest framing of the guarantee. With the default ``counters`` the exact
    top-``k`` is returned for any stream whose ``k``-th most frequent key clears
    the ``n / (counters + 1)`` threshold -- the usual case for skewed clickstream
    keys. The summary is a bounded-memory *approximation*: on an adversarial
    near-uniform stream a true heavy hitter can be evicted, so this guarantees the
    frequency-threshold property above, not unconditional exactness. Raise
    ``counters`` to tighten the guarantee (at ``counters >= distinct keys`` it is
    exact); the default is ``max(2 * k, k + 8)``, comfortably above ``k``.

    Ordering is deterministic: descending count, ties broken by ascending key.

    Worked example (hand-derived)
    -----------------------------
    ``keys = [a, a, a, b, b, c, a, c, c]``, ``k = 2`` with ``counters = 2``.
    Processing each key, with at most two counters:

    ===== ================ ========================================
    key   counters after   note
    ===== ================ ========================================
    a     {a: 1}           free counter taken
    a     {a: 2}           increment existing
    a     {a: 3}           increment existing
    b     {a: 3, b: 1}     second free counter taken
    b     {a: 3, b: 2}     increment existing
    c     {a: 2, b: 1}     full + new key -> decrement all
    a     {a: 3, b: 1}     increment existing
    c     {a: 2}           full + new key -> decrement all, drop b
    c     {a: 2, c: 1}     free counter taken
    ===== ================ ========================================

    Surviving candidates ``{a, c}``. The exact second pass gives true counts
    ``a=4, c=3``, so the top two is ``[(a, 4), (c, 3)]``.

    Parameters
    ----------
    keys:
        Sequence of hashable event keys (e.g. event names, page ids, user ids).
    k:
        Number of heavy hitters to return. Must be a positive integer.
    counters:
        Size of the Misra-Gries summary (number of tracked tallies). Must be
        >= 1 when given. Defaults to ``max(2 * k, k + 8)``.

    Returns
    -------
    list of (key, count)
        Up to ``k`` ``(key, exact_count)`` pairs, sorted by descending count then
        ascending key. Empty if ``keys`` is empty.

    Raises
    ------
    ValueError
        If ``k`` is not a positive integer, or ``counters`` is given and < 1.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if counters is None:
        capacity = max(2 * k, k + 8)
    elif not isinstance(counters, int) or counters < 1:
        raise ValueError("counters must be a positive integer when given.")
    else:
        capacity = counters

    # --- Misra-Gries pass: keep at most `capacity` counters -------------------
    tallies: dict[Hashable, int] = {}
    for key in keys:
        if key in tallies:
            tallies[key] += 1
        elif len(tallies) < capacity:
            tallies[key] = 1
        else:
            # No room: decrement every counter, drop those reaching zero.
            for tracked in list(tallies):
                tallies[tracked] -= 1
                if tallies[tracked] == 0:
                    del tallies[tracked]

    candidates: set[Hashable] = set(tallies)
    if not candidates:
        return []

    # --- Exact second pass over the surviving candidates ----------------------
    exact: dict[Hashable, int] = defaultdict(int)
    for key in keys:
        if key in candidates:
            exact[key] += 1

    ranked = sorted(exact.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]


def reorder_within_lateness(
    events: Sequence[tuple[float, Hashable]], allowed_lateness_s: float
) -> tuple[list[tuple[float, Hashable]], int]:
    """Re-order an out-of-order stream within an allowed-lateness bound.

    Events arrive in *arrival* order (the order of ``events``) but carry an
    event-time ``ts`` that may be out of order. A watermark tracks
    ``max_ts_seen - allowed_lateness_s`` (the same rule as
    :func:`clickstream.watermark.advance_watermark`). On each arrival the
    watermark is advanced first; an event whose ``ts`` is strictly below the new
    watermark is *too late to place* and is dropped (counted), otherwise it is
    buffered. Once buffered, any event whose ``ts`` is at or below the current
    watermark is *settled* and can be emitted in timestamp order. After the stream
    ends the watermark is pushed to ``+inf`` so the whole buffer drains in order.

    The emitted list is therefore globally sorted by ``ts`` (ties keep arrival
    order, a stable sort), and the returned count is the number of events that
    arrived too late to be ordered correctly.

    Worked example (hand-derived)
    -----------------------------
    Arrival order (ts, key), ``allowed_lateness_s = 2``::

        (10, a), (12, b), (11, c), (9, d), (13, e)

    Watermark = max_ts - 2:

    ===== ======== ========= =================================================
    ts    max_ts   watermark action
    ===== ======== ========= =================================================
    10    10       8         9 < ... no: ts 10 >= 8 -> buffer
    12    12       10        ts 12 >= 10 -> buffer
    11    11(*)    10        ts 11 >= 10 -> buffer  (*max stays 12, wm stays 10)
    9     12       10        ts 9 < 10 -> TOO LATE, drop (count 1)
    13    13       11        ts 13 >= 11 -> buffer
    ===== ======== ========= =================================================

    Buffer holds ts {10, 11, 12, 13}; draining at end emits them sorted:
    ``[(10, a), (11, c), (12, b), (13, e)]`` with ``dropped == 1``.

    Parameters
    ----------
    events:
        Sequence of ``(ts, key)`` pairs in *arrival* order.
    allowed_lateness_s:
        Grace period in seconds. Must be non-negative. ``0`` means an event is
        dropped only if it is strictly older than the maximum timestamp seen.

    Returns
    -------
    (ordered, dropped)
        ``ordered`` is the list of emitted ``(ts, key)`` pairs in non-decreasing
        ``ts`` order; ``dropped`` is the count of too-late events.

    Raises
    ------
    ValueError
        If ``allowed_lateness_s`` is negative.
    """
    if allowed_lateness_s < 0:
        raise ValueError("allowed_lateness_s must be non-negative.")

    buffer: list[tuple[int, float, Hashable]] = []
    ordered: list[tuple[float, Hashable]] = []
    dropped = 0
    watermark = float("-inf")
    max_ts = float("-inf")

    for arrival, (ts, key) in enumerate(events):
        if ts > max_ts:
            max_ts = ts
        watermark = max_ts - allowed_lateness_s
        if ts < watermark:
            dropped += 1
            continue
        # Carry arrival index so equal-ts events keep a stable, deterministic order.
        buffer.append((arrival, ts, key))

    # Drain everything in timestamp order (watermark -> +inf at end of stream).
    buffer.sort(key=lambda item: (item[1], item[0]))
    ordered = [(ts, key) for _arrival, ts, key in buffer]
    return ordered, dropped


def funnel_time_to_convert(
    user_events: Mapping[Hashable, Sequence[tuple[float, str]]],
    steps: Sequence[str],
) -> list[float | None]:
    """Median seconds between consecutive completed funnel steps.

    For each user, walk their time-ordered ``(ts, event)`` list and record, for
    each adjacent step pair ``steps[i] -> steps[i + 1]``, the time from the
    *first* occurrence of ``steps[i]`` (at or after the user reached step ``i``)
    to the *first* subsequent occurrence of ``steps[i + 1]``. Only users who
    completed both steps in order contribute a duration to that transition. The
    return is the median of the collected durations per transition.

    There are ``len(steps) - 1`` transitions. A transition with no completing
    user yields ``None`` (no median is defined).

    Median convention: the durations are sorted; for an odd count the middle
    value is returned, for an even count the mean of the two middle values.

    Worked example (hand-derived)
    -----------------------------
    ``steps = [view, cart, buy]``. Users (each list already time-sorted)::

        u1: (0, view), (10, cart), (30, buy)
        u2: (0, view), (20, cart)
        u3: (5, view), (5, cart),  (50, buy)

    view -> cart durations: u1 = 10-0 = 10, u2 = 20-0 = 20, u3 = 5-5 = 0.
        sorted [0, 10, 20] -> median 10.
    cart -> buy durations: u1 = 30-10 = 20, u3 = 50-5 = 45  (u2 never bought).
        sorted [20, 45] -> median (20+45)/2 = 32.5.

    => ``[10.0, 32.5]``.

    Parameters
    ----------
    user_events:
        Mapping ``user -> time-ordered list of (ts, event)`` pairs.
    steps:
        Ordered funnel step names. Must have at least two steps.

    Returns
    -------
    list of (float or None)
        Length ``len(steps) - 1``; element ``i`` is the median seconds for
        transition ``steps[i] -> steps[i + 1]``, or ``None`` if no user completed
        it.

    Raises
    ------
    ValueError
        If ``steps`` has fewer than two entries.
    """
    if len(steps) < 2:
        raise ValueError("steps must have at least two entries.")

    n_trans = len(steps) - 1
    durations: list[list[float]] = [[] for _ in range(n_trans)]

    for events in user_events.values():
        # Time of the first occurrence of each step as the user advances in order.
        step_idx = 0
        step_times: list[float | None] = [None] * len(steps)
        for ts, ev in events:
            if step_idx < len(steps) and ev == steps[step_idx]:
                step_times[step_idx] = ts
                step_idx += 1
        # Record a duration for every consecutive pair the user completed.
        for i in range(n_trans):
            t0 = step_times[i]
            t1 = step_times[i + 1]
            if t0 is not None and t1 is not None:
                durations[i].append(float(t1) - float(t0))

    return [_median(ds) for ds in durations]


def _median(values: Sequence[float]) -> float | None:
    """Median of ``values`` (mean of the two middles when even); ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def retention(
    user_events_with_ts: Mapping[Hashable, Sequence[float]], period_s: float
) -> list[float]:
    """Fraction of users retained across consecutive periods.

    Each user maps to a list of activity timestamps (epoch seconds). Time is
    bucketed into fixed periods of width ``period_s`` aligned to the epoch:
    ``period_index = floor(ts / period_s)``. A user is "active" in a period if any
    of their timestamps fall in it. For each pair of *consecutive* period indices
    ``(p, p + 1)`` spanned by the data, retention is::

        (# users active in BOTH p and p+1) / (# users active in p)

    The return is one fraction per consecutive period transition, from the first
    period that has any activity to the last, covering every integer period index
    in between (a fully empty period contributes a ``0.0`` retention from the
    previous period, since none of its predecessors' users returned).

    Worked example (hand-derived)
    -----------------------------
    ``period_s = 100``. Activity timestamps per user::

        u1: [10, 120]        -> active in periods 0, 1
        u2: [50, 150, 250]   -> active in periods 0, 1, 2
        u3: [30]             -> active in period 0 only
        u4: [220]            -> active in period 2 only

    Active sets: P0 = {u1, u2, u3}, P1 = {u1, u2}, P2 = {u2, u4}.

    Transition 0->1: both = {u1, u2} = 2, active in P0 = 3 -> 2/3 = 0.6667.
    Transition 1->2: both = {u2} = 1,     active in P1 = 2 -> 1/2 = 0.5.

    => ``[0.6667, 0.5]`` (rounded here for display; the function returns full
    precision).

    Parameters
    ----------
    user_events_with_ts:
        Mapping ``user -> list of activity timestamps`` (epoch seconds).
    period_s:
        Period width in seconds. Must be positive.

    Returns
    -------
    list of float
        Retention fractions, one per consecutive period transition across the
        full span of observed periods. Empty if there is no activity or only one
        period is occupied. A transition whose source period has no users yields
        ``0.0``.

    Raises
    ------
    ValueError
        If ``period_s`` is not positive.
    """
    if period_s <= 0:
        raise ValueError("period_s must be positive.")

    # period_index -> set of active users.
    active: dict[int, set[Hashable]] = defaultdict(set)
    for user, timestamps in user_events_with_ts.items():
        for ts in timestamps:
            period = int(ts // period_s)
            active[period].add(user)

    if not active:
        return []

    lo = min(active)
    hi = max(active)
    if hi == lo:
        return []

    fractions: list[float] = []
    for p in range(lo, hi):
        cur = active.get(p, set())
        nxt = active.get(p + 1, set())
        if not cur:
            fractions.append(0.0)
            continue
        retained = len(cur & nxt)
        fractions.append(retained / len(cur))
    return fractions
