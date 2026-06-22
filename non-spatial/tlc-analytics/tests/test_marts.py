"""Known-answer tests for the aggregated marts.

A tiny *already cleaned* frame (so ``trip_minutes`` and ``tip_pct`` are present)
with hand-counted answers.

Six trips:
  pickup            dow   hour  payment  fare  tip_pct  trip_minutes
  2023-01-02 09:00  Mon=0  9    card     10.0   0.20    10
  2023-01-02 09:30  Mon=0  9    card     20.0   0.10    20
  2023-01-03 18:00  Tue=1  18   cash     30.0   0.00    30
  2023-01-03 18:45  Tue=1  18   card     40.0   0.30    40
  2023-01-07 09:15  Sat=5  9    cash     50.0   0.00    50
  2023-01-07 23:00  Sat=5  23   card     10.0   0.40    10

Hand-derived:
  hourly_demand  -> hour 9: 3 trips, 18: 2 trips, 23: 1 trip
  demand_by_dow  -> Mon(0): 2, Tue(1): 2, Sat(5): 2
  tip_rate       -> card mean = (0.20+0.10+0.30+0.40)/4 = 0.25
                    cash mean = (0.00+0.00)/2           = 0.00
  fare_summary   -> trips 6; mean_fare (10+20+30+40+50+10)/6 = 160/6 = 26.6667
                    median_fare = mean(20,30) = 25.0
                    mean_trip_minutes = (10+20+30+40+50+10)/6 = 160/6 = 26.6667
"""

from __future__ import annotations

import pandas as pd
import pytest

from tlc.marts import (
    demand_by_dow,
    fare_summary,
    hourly_demand,
    tip_rate_by_payment,
)


def _clean_frame() -> pd.DataFrame:
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
            "payment_type": ["card", "card", "cash", "card", "cash", "card"],
            "fare_amount": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0],
            "tip_pct": [0.20, 0.10, 0.00, 0.30, 0.00, 0.40],
            "trip_minutes": [10.0, 20.0, 30.0, 40.0, 50.0, 10.0],
        }
    )


def test_hourly_demand() -> None:
    out = hourly_demand(_clean_frame())
    got = dict(zip(out["hour"], out["trips"], strict=True))
    assert got == {9: 3, 18: 2, 23: 1}


def test_demand_by_dow() -> None:
    out = demand_by_dow(_clean_frame())
    got = dict(zip(out["dow"], out["trips"], strict=True))
    assert got == {0: 2, 1: 2, 5: 2}


def test_tip_rate_by_payment() -> None:
    out = tip_rate_by_payment(_clean_frame())
    got = dict(zip(out["payment_type"], out["mean_tip_pct"], strict=True))
    assert got["card"] == pytest.approx(0.25)
    assert got["cash"] == pytest.approx(0.00)


def test_fare_summary() -> None:
    out = fare_summary(_clean_frame())
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["trips"]) == 6
    assert row["mean_fare"] == pytest.approx(160.0 / 6.0)
    assert row["median_fare"] == pytest.approx(25.0)
    assert row["mean_trip_minutes"] == pytest.approx(160.0 / 6.0)
