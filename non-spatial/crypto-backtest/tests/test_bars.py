"""Known-answer tests for tick -> OHLCV resampling.

A tiny, hand-checkable tick set is resampled to 1-minute bars and every
open/high/low/close/volume is compared against a value computed by hand.

Ticks (timestamp, price, size), left-closed 1-minute bars:

  09:00:10  100  1  ->  bar 09:00
  09:00:30  105  2  ->  bar 09:00
  09:00:50  102  1  ->  bar 09:00
  09:01:20  108  3  ->  bar 09:01

Bar 09:00 : open=100 (first), high=105, low=100, close=102 (last), volume=4.
Bar 09:01 : open=high=low=close=108, volume=3.
"""

from __future__ import annotations

import pandas as pd

from backtest.bars import resample_ohlcv

TICKS = pd.DataFrame(
    {
        "ts": pd.to_datetime(
            [
                "2024-01-01 09:00:10",
                "2024-01-01 09:00:30",
                "2024-01-01 09:00:50",
                "2024-01-01 09:01:20",
            ]
        ),
        "price": [100.0, 105.0, 102.0, 108.0],
        "size": [1.0, 2.0, 1.0, 3.0],
    }
)


def test_resample_ohlcv_known_answer() -> None:
    bars = resample_ohlcv(TICKS, "1min", ts="ts")

    assert list(bars.index) == [
        pd.Timestamp("2024-01-01 09:00:00"),
        pd.Timestamp("2024-01-01 09:01:00"),
    ]

    b0 = bars.iloc[0]
    assert b0["open"] == 100.0
    assert b0["high"] == 105.0
    assert b0["low"] == 100.0
    assert b0["close"] == 102.0
    assert b0["volume"] == 4.0

    b1 = bars.iloc[1]
    assert b1["open"] == 108.0
    assert b1["high"] == 108.0
    assert b1["low"] == 108.0
    assert b1["close"] == 108.0
    assert b1["volume"] == 3.0


def test_resample_accepts_datetime_index() -> None:
    indexed = TICKS.set_index("ts")
    bars = resample_ohlcv(indexed, "1min")
    assert bars["close"].tolist() == [102.0, 108.0]


def test_empty_intervals_are_dropped() -> None:
    # A gap at 09:01 (no trades) must not produce a synthetic bar.
    sparse = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01 09:00:10", "2024-01-01 09:02:10"]),
            "price": [100.0, 110.0],
            "size": [1.0, 1.0],
        }
    )
    bars = resample_ohlcv(sparse, "1min", ts="ts")
    assert list(bars.index) == [
        pd.Timestamp("2024-01-01 09:00:00"),
        pd.Timestamp("2024-01-01 09:02:00"),
    ]
