"""Watermarks and late-event handling for event-time streaming (pure Python).

In an event-time pipeline the *watermark* is the engine's belief about how far
event time has progressed: a claim that "no more events with timestamp <=
watermark are expected". Events that arrive with a timestamp at or before the
current watermark are *late* and would otherwise be dropped or update an already
emitted window.

These two functions are the deterministic core of that logic, with no
third-party dependency, so they are always importable and unit-tested with
hand-derived values. The Spark Structured Streaming engine in
:mod:`clickstream.pipeline` configures an equivalent watermark via
``withWatermark``; this module is the reference an analyst can reason about.
"""

from __future__ import annotations


def is_late(event_ts: float, watermark_ts: float) -> bool:
    """Return whether an event is late relative to the current watermark.

    An event is late when its event-time timestamp is at or before the
    watermark, i.e. ``event_ts <= watermark_ts``. An event exactly on the
    watermark is treated as late, matching the "no more events <= watermark"
    contract.

    Parameters
    ----------
    event_ts:
        Event-time timestamp of the event (epoch seconds).
    watermark_ts:
        Current watermark (epoch seconds).

    Returns
    -------
    bool
        ``True`` if the event is late and should be handled as such.
    """
    return event_ts <= watermark_ts


def advance_watermark(
    current_wm: float, event_ts: float, allowed_lateness_s: float
) -> float:
    """Advance the watermark given a newly observed event timestamp.

    The watermark tracks the maximum event timestamp seen so far minus the
    allowed lateness: ``watermark = max_event_ts - allowed_lateness_s``. It is
    monotonically non-decreasing, so a late or out-of-order event (whose
    timestamp is below the running maximum) never pulls the watermark back.

    Parameters
    ----------
    current_wm:
        The current watermark (epoch seconds).
    event_ts:
        The event-time timestamp of the newly observed event (epoch seconds).
    allowed_lateness_s:
        Grace period in seconds: how far behind the maximum observed event time
        the watermark is held, so that events arriving within this delay are
        still accepted. Must be non-negative.

    Returns
    -------
    float
        The new watermark, never less than ``current_wm``.

    Raises
    ------
    ValueError
        If ``allowed_lateness_s`` is negative.
    """
    if allowed_lateness_s < 0:
        raise ValueError("allowed_lateness_s must be non-negative.")
    candidate = event_ts - allowed_lateness_s
    return candidate if candidate > current_wm else current_wm
