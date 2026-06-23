"""Transforms: pure-pandas marts that mirror the dbt models.

These two functions are the pandas analogue of the dbt marts in ``transform/``:

* :func:`daily_summary` is the clean daily grain (one row per station/day), the
  mirror of ``daily_weather.sql``.
* :func:`monthly_summary` rolls the daily grain up to one row per station/month
  with temperature stats, total precipitation, a rain-day count and a record
  count — the mirror of ``monthly_weather.sql``.

Keeping the marts in pure pandas means the aggregation logic is unit-tested with
hand-derived known answers, independent of DuckDB or dbt.
"""

from __future__ import annotations

import pandas as pd

#: A day counts as a "rain day" when precipitation reaches this many mm.
RAIN_DAY_MM = 1.0


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return the clean daily table, sorted by (station, date).

    One row per station/day with the canonical columns. This is intentionally
    thin — it mirrors the staging-to-mart daily model — and assumes the frame has
    already passed :func:`weatherpipe.validate.validate_weather`.
    """
    cols = ["station", "date", "tmin_c", "tmax_c", "tmean_c", "precip_mm"]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["station", "date"]).reset_index(drop=True)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily grain to one row per station/month.

    Columns:

    * ``station``, ``year``, ``month`` — the grain.
    * ``tmean_mean`` — mean of daily mean temperature.
    * ``tmin_min``   — coldest daily minimum.
    * ``tmax_max``   — hottest daily maximum.
    * ``precip_total_mm`` — total precipitation over the month.
    * ``rain_days`` — count of days with ``precip_mm >= RAIN_DAY_MM``.
    * ``record_count`` — number of daily rows in the month.

    Rows are sorted by (station, year, month). The result is hand-derivable, which
    is exactly what ``tests/test_transform.py`` pins.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.month
    work["is_rain_day"] = work["precip_mm"] >= RAIN_DAY_MM

    grouped = work.groupby(["station", "year", "month"], as_index=False).agg(
        tmean_mean=("tmean_c", "mean"),
        tmin_min=("tmin_c", "min"),
        tmax_max=("tmax_c", "max"),
        precip_total_mm=("precip_mm", "sum"),
        rain_days=("is_rain_day", "sum"),
        record_count=("date", "size"),
    )
    grouped["rain_days"] = grouped["rain_days"].astype(int)
    grouped["record_count"] = grouped["record_count"].astype(int)
    return grouped.sort_values(["station", "year", "month"]).reset_index(drop=True)
