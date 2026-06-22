"""Known-answer test for the pandas events-per-minute aggregation.

Hand-derived expectation, numpy/pandas only (no streaming engine).

Worked example:

  Six events at epoch seconds ts = [0, 10, 59, 60, 125, 130].
  Minute bucket = floor(ts/60)*60:
    0   -> 0
    10  -> 0
    59  -> 0
    60  -> 60
    125 -> 120
    130 -> 120
  Counts per bucket: {0: 3, 60: 1, 120: 2}, sorted by minute.
"""

from __future__ import annotations

import pandas as pd
import pytest

from clickstream import events_per_minute


def test_events_per_minute_known() -> None:
    """Hand-counted minute buckets -> {0: 3, 60: 1, 120: 2}."""
    df = pd.DataFrame({"ts": [0, 10, 59, 60, 125, 130]})
    result = events_per_minute(df)

    assert list(result.index) == [0, 60, 120]
    assert list(result.to_numpy()) == [3, 1, 2]
    assert result.index.name == "minute"
    assert result.name == "count"


def test_events_per_minute_requires_ts_column() -> None:
    with pytest.raises(ValueError):
        events_per_minute(pd.DataFrame({"other": [1, 2, 3]}))


def test_events_per_minute_single_event() -> None:
    """A single event lands in exactly one minute bucket."""
    result = events_per_minute(pd.DataFrame({"ts": [125]}))
    assert list(result.index) == [120]
    assert list(result.to_numpy()) == [1]


def test_events_per_minute_empty() -> None:
    """No rows -> an empty count series (no buckets)."""
    result = events_per_minute(pd.DataFrame({"ts": []}))
    assert len(result) == 0
