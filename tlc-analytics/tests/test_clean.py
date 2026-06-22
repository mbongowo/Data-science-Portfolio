"""Known-answer tests for clean_trips.

A tiny frame with deliberately planted bad rows. Each bad row violates exactly
one predicate, so a correct cleaner drops those four and keeps the rest, and the
derived columns (``trip_minutes``, ``tip_pct``) take hand-computed values.

Rows (index, why):
  0  good          fare 10, tip 2, 2 pax, 09:00 -> 09:15  (15 min, tip_pct 0.20)
  1  good          fare 20, tip 5, 1 pax, 18:30 -> 18:40  (10 min, tip_pct 0.25)
  2  BAD fare      fare 0   -> dropped (non-positive fare)
  3  BAD pax       fare 8, 0 passengers -> dropped
  4  BAD duration  dropoff before pickup -> dropped (non-positive duration)
  5  BAD fare cap  fare 9000 -> dropped (above the 500 cap)
"""

from __future__ import annotations

import pandas as pd
import pytest

from tlc.clean import clean_trips


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_datetime": [
                "2023-01-02 09:00:00",
                "2023-01-02 18:30:00",
                "2023-01-02 10:00:00",
                "2023-01-02 11:00:00",
                "2023-01-02 12:30:00",
                "2023-01-02 13:00:00",
            ],
            "dropoff_datetime": [
                "2023-01-02 09:15:00",
                "2023-01-02 18:40:00",
                "2023-01-02 10:20:00",
                "2023-01-02 11:10:00",
                "2023-01-02 12:00:00",  # before pickup
                "2023-01-02 13:30:00",
            ],
            "fare_amount": [10.0, 20.0, 0.0, 8.0, 12.0, 9000.0],
            "tip_amount": [2.0, 5.0, 1.0, 1.0, 1.0, 100.0],
            "passenger_count": [2, 1, 1, 0, 1, 1],
        }
    )


def test_drops_exactly_the_planted_bad_rows() -> None:
    out = clean_trips(_frame())
    # Only rows 0 and 1 survive.
    assert len(out) == 2
    assert out["fare_amount"].tolist() == [10.0, 20.0]


def test_derived_columns_are_correct() -> None:
    out = clean_trips(_frame())
    assert out["trip_minutes"].tolist() == [15.0, 10.0]
    assert out["tip_pct"].tolist() == pytest.approx([0.20, 0.25])


def test_fare_cap_is_configurable() -> None:
    # Raise the cap above 9000 and the big-fare row (which is otherwise valid)
    # is kept: 09:00->09:15 style rows 0,1 plus the 13:00->13:30 row 5.
    out = clean_trips(_frame(), fare_cap=10_000.0)
    assert len(out) == 3
    assert 9000.0 in out["fare_amount"].tolist()


def test_missing_column_raises() -> None:
    bad = _frame().drop(columns=["tip_amount"])
    with pytest.raises(KeyError):
        clean_trips(bad)
