"""Known-answer tests for the data-integrity checks.

* find_gaps on 1-minute timestamps [09:00, 09:01, 09:03] reports the single
  missing 09:02.
* find_duplicates on [09:00, 09:01, 09:01] reports 09:01 once.
* summarize_integrity reports the bar count, gap count, duplicate count, and a
  coverage fraction (observed unique / expected on the grid).
"""

from __future__ import annotations

import pandas as pd

from backtest.integrity import find_duplicates, find_gaps, summarize_integrity


def test_find_gaps_single_missing_minute() -> None:
    idx = pd.to_datetime(["2024-01-01 09:00", "2024-01-01 09:01", "2024-01-01 09:03"])
    gaps = find_gaps(idx, "1min")
    assert list(gaps) == [pd.Timestamp("2024-01-01 09:02")]


def test_find_gaps_none_when_complete() -> None:
    idx = pd.date_range("2024-01-01 09:00", periods=5, freq="1min")
    assert len(find_gaps(idx, "1min")) == 0


def test_find_duplicates() -> None:
    idx = pd.to_datetime(["2024-01-01 09:00", "2024-01-01 09:01", "2024-01-01 09:01"])
    dups = find_duplicates(idx)
    assert list(dups) == [pd.Timestamp("2024-01-01 09:01")]


def test_find_duplicates_none() -> None:
    idx = pd.date_range("2024-01-01 09:00", periods=3, freq="1min")
    assert len(find_duplicates(idx)) == 0


def test_summarize_integrity_counts() -> None:
    idx = pd.to_datetime(["2024-01-01 09:00", "2024-01-01 09:01", "2024-01-01 09:03"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    summary = summarize_integrity(df, "1min")

    assert summary["n_bars"] == 3
    assert summary["n_gaps"] == 1
    assert summary["n_duplicates"] == 0
    # Expected grid 09:00..09:03 = 4 bars; observed unique = 3 -> 0.75.
    assert summary["coverage"] == 0.75


def test_summarize_integrity_complete() -> None:
    idx = pd.date_range("2024-01-01 09:00", periods=10, freq="1min")
    df = pd.DataFrame({"close": range(10)}, index=idx)
    summary = summarize_integrity(df, "1min")
    assert summary["coverage"] == 1.0
    assert summary["n_gaps"] == 0
