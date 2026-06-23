"""Tumbling-window aggregation and small stream helpers (pure Python).

This module is part of the interpretation-critical core. It has no third-party
dependency beyond the standard library, so it is always importable and is the
basis of known-answer unit tests. The streaming engine (Kafka + Spark) that
feeds these in production lives in :mod:`aqstream.stream` and is imported lazily
there, never here.

Timestamps are epoch *seconds* throughout. Window boundaries are aligned to the
epoch: a window of width ``window_s`` starting at ``t`` covers
``[t, t + window_s)`` (half-open), so a reading exactly on a boundary belongs to
the later window. That convention is what the tests check.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence

_AGGREGATORS = ("mean", "max", "min", "count")


def tumbling_aggregate(
    readings: Sequence[dict],
    window_s: float,
    value_key: str,
    agg: str = "mean",
) -> dict[tuple[float, Hashable], float]:
    """Aggregate a reading stream into fixed windows, per (window_start, station).

    Each reading is a mapping with at least a ``ts`` (epoch seconds), a
    ``station`` identifier, and the numeric field named by ``value_key``. A
    reading is assigned to the window whose start is
    ``floor(ts / window_s) * window_s`` (half-open ``[start, start + window_s)``),
    and aggregated per ``(window_start, station)``.

    Parameters
    ----------
    readings:
        Sequence of reading dicts. Each must contain ``ts``, ``station`` and
        ``value_key``.
    window_s:
        Window width in seconds. Must be positive.
    value_key:
        Name of the numeric field to aggregate (e.g. ``"pm25"``).
    agg:
        One of ``"mean"``, ``"max"``, ``"min"``, ``"count"``.

    Returns
    -------
    dict
        Mapping ``{(window_start, station): aggregate}``. ``"count"`` returns an
        integer count as a float-compatible value; the others return the
        aggregate of the values.

    Raises
    ------
    ValueError
        If ``window_s`` is not positive, ``agg`` is unknown, or a reading is
        missing a required field.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive.")
    if agg not in _AGGREGATORS:
        raise ValueError(f"agg must be one of {_AGGREGATORS}, got {agg!r}.")

    buckets: dict[tuple[float, Hashable], list[float]] = defaultdict(list)
    for r in readings:
        for field in ("ts", "station"):
            if field not in r:
                raise ValueError(f"reading is missing required field {field!r}.")
        if agg != "count" and value_key not in r:
            raise ValueError(f"reading is missing value field {value_key!r}.")

        ts = float(r["ts"])
        start = math.floor(ts / window_s) * window_s
        key = (start, r["station"])
        if agg == "count":
            buckets[key].append(1.0)
        else:
            buckets[key].append(float(r[value_key]))

    out: dict[tuple[float, Hashable], float] = {}
    for key, values in buckets.items():
        if agg == "mean":
            out[key] = sum(values) / len(values)
        elif agg == "max":
            out[key] = max(values)
        elif agg == "min":
            out[key] = min(values)
        else:  # count
            out[key] = float(len(values))
    return out


def rolling_mean(values: Sequence[float], n: int) -> list[float | None]:
    """Trailing rolling mean of width ``n``.

    Element ``i`` is the mean of ``values[i - n + 1 .. i]`` once at least ``n``
    values are available; positions before the window is full are ``None``. The
    output has the same length as ``values``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    out: list[float | None] = []
    running = 0.0
    for i, v in enumerate(values):
        running += float(v)
        if i >= n:
            running -= float(values[i - n])
        if i >= n - 1:
            out.append(running / n)
        else:
            out.append(None)
    return out


def dedupe(readings: Sequence[dict], key: str) -> list[dict]:
    """Drop duplicate readings, keeping the first occurrence per ``key`` value.

    Order is preserved. ``key`` is the field whose value identifies a reading
    (e.g. a composite ``"id"`` or a ``(station, ts)`` you have pre-computed).

    Raises
    ------
    ValueError
        If a reading is missing ``key``.
    """
    seen: set[Hashable] = set()
    out: list[dict] = []
    for r in readings:
        if key not in r:
            raise ValueError(f"reading is missing dedupe key {key!r}.")
        marker = r[key]
        if marker in seen:
            continue
        seen.add(marker)
        out.append(r)
    return out
