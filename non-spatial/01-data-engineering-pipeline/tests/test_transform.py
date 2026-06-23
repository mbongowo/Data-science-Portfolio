"""Known-answer tests for the monthly_summary mart on a hand-laid frame."""

from __future__ import annotations

import pandas as pd

from weatherpipe.transform import daily_summary, monthly_summary


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Three days for one station in 2023-01, with hand-computable aggregates:
#   tmean_c       : 26, 28, 30        -> mean 28.0
#   tmin_c        : 20, 22, 21        -> min  20.0
#   tmax_c        : 31, 35, 33        -> max  35.0
#   precip_mm     : 0.0, 1.0, 12.5    -> total 13.5, rain_days 2 (>= 1.0)
_ROWS = [
    {"station": "Douala", "date": "2023-01-01", "tmin_c": 20.0, "tmax_c": 31.0,
     "tmean_c": 26.0, "precip_mm": 0.0},
    {"station": "Douala", "date": "2023-01-02", "tmin_c": 22.0, "tmax_c": 35.0,
     "tmean_c": 28.0, "precip_mm": 1.0},
    {"station": "Douala", "date": "2023-01-03", "tmin_c": 21.0, "tmax_c": 33.0,
     "tmean_c": 30.0, "precip_mm": 12.5},
]


def test_monthly_summary_known_answer() -> None:
    out = monthly_summary(_frame(_ROWS))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["station"] == "Douala"
    assert int(row["year"]) == 2023
    assert int(row["month"]) == 1
    assert row["tmean_mean"] == 28.0
    assert row["tmin_min"] == 20.0
    assert row["tmax_max"] == 35.0
    assert row["precip_total_mm"] == 13.5
    assert int(row["rain_days"]) == 2
    assert int(row["record_count"]) == 3


def test_monthly_summary_splits_by_station_and_month() -> None:
    rows = _ROWS + [
        {"station": "Douala", "date": "2023-02-01", "tmin_c": 23.0, "tmax_c": 34.0,
         "tmean_c": 29.0, "precip_mm": 0.5},
        {"station": "Maroua", "date": "2023-01-01", "tmin_c": 18.0, "tmax_c": 36.0,
         "tmean_c": 27.0, "precip_mm": 0.0},
    ]
    out = monthly_summary(_frame(rows))
    # Three groups: Douala/Jan, Douala/Feb, Maroua/Jan. Sorted by station/year/month.
    assert len(out) == 3
    assert list(out["station"]) == ["Douala", "Douala", "Maroua"]
    assert list(out["month"]) == [1, 2, 1]
    # Maroua/Jan has a single dry day: 0 rain days, total 0.0.
    maroua = out[out["station"] == "Maroua"].iloc[0]
    assert int(maroua["rain_days"]) == 0
    assert maroua["precip_total_mm"] == 0.0
    assert int(maroua["record_count"]) == 1


def test_daily_summary_sorts_and_keeps_columns() -> None:
    shuffled = _frame(list(reversed(_ROWS)))
    out = daily_summary(shuffled)
    assert list(out.columns) == [
        "station", "date", "tmin_c", "tmax_c", "tmean_c", "precip_mm",
    ]
    assert list(out["date"]) == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-03"),
    ]
