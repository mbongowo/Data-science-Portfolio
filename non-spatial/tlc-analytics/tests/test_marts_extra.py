"""Known-answer + edge-case tests for the added marts.

The three new marts (``trip_duration_buckets``, ``revenue_by_day``,
``anomaly_flags``) are pinned on a tiny hand-derived frame, with edge cases for
the empty frame and the all-identical (zero-IQR) case.

Hand-derived reference (six already-cleaned trips):

  pickup            fare  tip   trip_minutes
  2023-01-02 09:00   10    2     3      -> bucket 0-5
  2023-01-02 09:30   20    1     7      -> bucket 5-10
  2023-01-03 18:00   30    0    15      -> bucket 10-20
  2023-01-03 18:45   40    4    25      -> bucket 20-30
  2023-01-07 09:15   50    0    45      -> bucket 30-60
  2023-01-07 23:00   10    4    90      -> bucket 60+

  trip_duration_buckets -> exactly one trip per band: each of the six labels = 1
  revenue_by_day        -> 2023-01-02: (10+2)+(20+1)          = 33
                           2023-01-03: (30+0)+(40+4)          = 74
                           2023-01-07: (50+0)+(10+4)          = 64
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tlc.marts import (
    DURATION_BIN_LABELS,
    anomaly_flags,
    revenue_by_day,
    trip_duration_buckets,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                [
                    "2023-01-02 09:00:00",
                    "2023-01-02 09:30:00",
                    "2023-01-03 18:00:00",
                    "2023-01-03 18:45:00",
                    "2023-01-07 09:15:00",
                    "2023-01-07 23:00:00",
                ]
            ),
            "fare_amount": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0],
            "tip_amount": [2.0, 1.0, 0.0, 4.0, 0.0, 4.0],
            "trip_minutes": [3.0, 7.0, 15.0, 25.0, 45.0, 90.0],
        }
    )


# --- trip_duration_buckets -------------------------------------------------


def test_trip_duration_buckets_one_per_band() -> None:
    out = trip_duration_buckets(_frame())
    # All six bands present, in order, exactly one trip each.
    assert out["bucket"].tolist() == list(DURATION_BIN_LABELS)
    assert out["trips"].tolist() == [1, 1, 1, 1, 1, 1]


def test_trip_duration_buckets_left_closed_right_open() -> None:
    # A trip of exactly 5.0 minutes lands in 5-10, not 0-5.
    df = pd.DataFrame({"trip_minutes": [5.0]})
    out = trip_duration_buckets(df)
    got = dict(zip(out["bucket"], out["trips"], strict=True))
    assert got["0-5"] == 0
    assert got["5-10"] == 1


def test_trip_duration_buckets_empty_zero_filled() -> None:
    empty = pd.DataFrame({"trip_minutes": pd.Series([], dtype=float)})
    out = trip_duration_buckets(empty)
    # Every band still present, all zero.
    assert out["bucket"].tolist() == list(DURATION_BIN_LABELS)
    assert out["trips"].tolist() == [0, 0, 0, 0, 0, 0]


# --- revenue_by_day --------------------------------------------------------


def test_revenue_by_day_known_answer() -> None:
    out = revenue_by_day(_frame())
    got = {
        str(pd.Timestamp(d).date()): r
        for d, r in zip(out["day"], out["revenue"], strict=True)
    }
    assert got == pytest.approx(
        {"2023-01-02": 33.0, "2023-01-03": 74.0, "2023-01-07": 64.0}
    )
    # Sorted ascending by day.
    days = [str(pd.Timestamp(d).date()) for d in out["day"]]
    assert days == sorted(days)


def test_revenue_by_day_single_day() -> None:
    df = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                ["2023-01-02 01:00:00", "2023-01-02 23:00:00"]
            ),
            "fare_amount": [10.0, 5.0],
            "tip_amount": [1.0, 0.0],
        }
    )
    out = revenue_by_day(df)
    assert len(out) == 1
    assert out["revenue"].iloc[0] == pytest.approx(16.0)


# --- anomaly_flags ---------------------------------------------------------


def test_anomaly_flags_detects_high_outlier() -> None:
    # Seven tight values plus one far-out fare; the outlier must be flagged and
    # nothing else. Q1=10, Q3=10 here would be zero-IQR, so spread the values.
    fares = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1000.0]
    trip = [5.0] * 8  # identical -> zero IQR -> never flagged
    df = pd.DataFrame({"fare_amount": fares, "trip_minutes": trip})
    out = anomaly_flags(df)
    assert out["fare_outlier"].tolist() == [False] * 7 + [True]
    assert out["trip_minutes_outlier"].tolist() == [False] * 8
    assert out["is_outlier"].tolist() == [False] * 7 + [True]
    assert len(out) == len(df)


def test_anomaly_flags_all_same_zero_iqr() -> None:
    # All-identical values: IQR is 0, no row may be flagged.
    df = pd.DataFrame({"fare_amount": [12.0] * 5, "trip_minutes": [8.0] * 5})
    out = anomaly_flags(df)
    assert not out["fare_outlier"].any()
    assert not out["trip_minutes_outlier"].any()
    assert not out["is_outlier"].any()


def test_anomaly_flags_empty_frame() -> None:
    df = pd.DataFrame({"fare_amount": pd.Series([], dtype=float),
                       "trip_minutes": pd.Series([], dtype=float)})
    out = anomaly_flags(df)
    assert len(out) == 0
    for col in ("fare_outlier", "trip_minutes_outlier", "is_outlier"):
        assert out[col].dtype == np.bool_


def test_anomaly_flags_matches_manual_tukey_fence() -> None:
    # Hand-derive the fence: values 1..9 plus a 100. With numpy linear quantiles
    # on the 10 sorted values, Q1 sits at position 0.25*9 = 2.25 ->
    # 3 + 0.25*(4-3) = 3.25, and Q3 at 0.75*9 = 6.75 -> 7 + 0.75*(8-7) = 7.75.
    # IQR = 4.5, upper fence = 7.75 + 1.5*4.5 = 14.5; only 100 exceeds it.
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    q1 = np.quantile(vals, 0.25)
    q3 = np.quantile(vals, 0.75)
    assert q1 == pytest.approx(3.25)
    assert q3 == pytest.approx(7.75)
    df = pd.DataFrame({"fare_amount": vals, "trip_minutes": [5.0] * 10})
    out = anomaly_flags(df)
    assert out["fare_outlier"].tolist() == [False] * 9 + [True]
