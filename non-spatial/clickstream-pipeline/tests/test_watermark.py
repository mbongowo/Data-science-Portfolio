"""Known-answer tests for the event-time watermark logic.

Hand-derived expectations, no streaming engine required.

Worked examples:

1. is_late(event_ts, watermark_ts) is event_ts <= watermark_ts.
     is_late(90, 100)  -> True  (90 is before the watermark)
     is_late(100, 100) -> True  (exactly on the watermark counts as late)
     is_late(110, 100) -> False (110 is ahead of the watermark)

2. advance_watermark(current_wm, event_ts, allowed_lateness_s)
   = max(current_wm, event_ts - allowed_lateness_s).
     start wm = 0, allowed lateness = 5.
     event 100 -> max(0, 100-5)  = 95.
     event 103 -> max(95, 103-5) = 98.
     event  60 -> max(98, 60-5)  = 98  (late event does not pull it back).
"""

from __future__ import annotations

import pytest

from clickstream import advance_watermark, is_late


def test_is_late_before_watermark() -> None:
    assert is_late(90.0, 100.0) is True


def test_is_late_exactly_on_watermark_is_late() -> None:
    """An event exactly on the watermark is treated as late."""
    assert is_late(100.0, 100.0) is True


def test_is_late_ahead_of_watermark() -> None:
    assert is_late(110.0, 100.0) is False


def test_advance_watermark_progresses() -> None:
    """event_ts - allowed_lateness advances the watermark when it is ahead."""
    assert advance_watermark(0.0, 100.0, 5.0) == 95.0


def test_advance_watermark_monotone() -> None:
    """A late/out-of-order event must not move the watermark backward."""
    wm = advance_watermark(0.0, 100.0, 5.0)  # 95
    wm = advance_watermark(wm, 103.0, 5.0)  # 98
    wm = advance_watermark(wm, 60.0, 5.0)  # stays 98
    assert wm == 98.0


def test_advance_watermark_rejects_negative_lateness() -> None:
    with pytest.raises(ValueError):
        advance_watermark(0.0, 100.0, -1.0)


def test_advance_watermark_zero_lateness() -> None:
    """With zero allowed lateness the watermark equals the max event time."""
    assert advance_watermark(0.0, 100.0, 0.0) == 100.0


def test_advance_watermark_equal_event_does_not_regress() -> None:
    """Re-observing the current max keeps the watermark put (non-decreasing)."""
    wm = advance_watermark(0.0, 100.0, 5.0)  # 95
    assert advance_watermark(wm, 100.0, 5.0) == 95.0
