"""Aggregated marts for cleaned TLC trips (pure pandas, no engine deps).

These four functions are the *reference* definitions of the analytics workload.
Each takes a cleaned frame (the output of :func:`tlc.clean.clean_trips`, i.e.
with ``trip_minutes`` and ``tip_pct`` present) and returns a small aggregated
DataFrame. The engine wrappers in ``engines.py`` express the identical logic in
SQL / Spark; these pandas versions are the source of truth and the basis of the
known-answer tests.

The aggregations:

* :func:`hourly_demand`         — trip counts by pickup hour (0-23).
* :func:`demand_by_dow`         — trip counts by day of week (0=Mon .. 6=Sun).
* :func:`tip_rate_by_payment`   — mean ``tip_pct`` by payment type.
* :func:`fare_summary`          — count, mean/median fare and mean trip duration.
* :func:`trip_duration_buckets` — trip counts binned by trip-minute band.
* :func:`revenue_by_day`        — summed fare + tip revenue per calendar day.
* :func:`anomaly_flags`         — IQR-based outlier flags on fare and duration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PICKUP_COL = "pickup_datetime"
FARE_COL = "fare_amount"
PAYMENT_COL = "payment_type"
TIP_COL = "tip_amount"
TRIP_MINUTES_COL = "trip_minutes"

#: Right-open trip-minute bands used by :func:`trip_duration_buckets`. The final
#: band is closed on the right at infinity (``60+``).
DURATION_BIN_EDGES = (0.0, 5.0, 10.0, 20.0, 30.0, 60.0, float("inf"))
DURATION_BIN_LABELS = ("0-5", "5-10", "10-20", "20-30", "30-60", "60+")


def hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips by pickup hour of day.

    Returns a frame with columns ``hour`` (0-23) and ``trips``, sorted by hour.
    Hours with no trips are simply absent (no zero-fill).
    """
    hour = pd.to_datetime(df[PICKUP_COL]).dt.hour
    counts = hour.value_counts().sort_index()
    return pd.DataFrame({"hour": counts.index.to_numpy(), "trips": counts.to_numpy()})


def demand_by_dow(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips by day of week.

    Day of week follows pandas' convention: ``0`` is Monday and ``6`` is Sunday.
    Returns a frame with columns ``dow`` and ``trips``, sorted by ``dow``.
    """
    dow = pd.to_datetime(df[PICKUP_COL]).dt.dayofweek
    counts = dow.value_counts().sort_index()
    return pd.DataFrame({"dow": counts.index.to_numpy(), "trips": counts.to_numpy()})


def tip_rate_by_payment(df: pd.DataFrame) -> pd.DataFrame:
    """Mean tip percentage by payment type.

    Returns a frame with columns ``payment_type`` and ``mean_tip_pct``, sorted
    by payment type. ``mean_tip_pct`` is the mean of the per-trip ``tip_pct``
    column (a fraction of the fare), grouped by payment type.
    """
    grouped = df.groupby(PAYMENT_COL, sort=True)["tip_pct"].mean()
    return pd.DataFrame(
        {
            "payment_type": grouped.index.to_numpy(),
            "mean_tip_pct": grouped.to_numpy(),
        }
    )


def fare_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise fares and trip duration over the whole frame.

    Returns a one-row frame with columns ``trips`` (row count), ``mean_fare``,
    ``median_fare`` and ``mean_trip_minutes``.
    """
    return pd.DataFrame(
        {
            "trips": [int(len(df))],
            "mean_fare": [float(df[FARE_COL].mean())],
            "median_fare": [float(df[FARE_COL].median())],
            "mean_trip_minutes": [float(df["trip_minutes"].mean())],
        }
    )


def trip_duration_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips by binned trip duration.

    Bins ``trip_minutes`` into the right-open bands defined by
    :data:`DURATION_BIN_EDGES` (``0-5``, ``5-10``, ``10-20``, ``20-30``,
    ``30-60``, ``60+`` minutes). Each band is left-closed / right-open
    (``[lo, hi)``); a trip of exactly 5 minutes lands in ``5-10``.

    Returns a frame with columns ``bucket`` (ordered string label) and
    ``trips``, with **every** band present (zero-filled) and in band order, so
    the distribution shape is stable across inputs.
    """
    cut = pd.cut(
        df[TRIP_MINUTES_COL],
        bins=list(DURATION_BIN_EDGES),
        labels=list(DURATION_BIN_LABELS),
        right=False,
    )
    counts = cut.value_counts().reindex(list(DURATION_BIN_LABELS), fill_value=0)
    return pd.DataFrame(
        {
            "bucket": list(DURATION_BIN_LABELS),
            "trips": counts.to_numpy().astype(int),
        }
    )


def revenue_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Sum trip revenue (fare + tip) per calendar day.

    Revenue per trip is ``fare_amount + tip_amount``. Returns a frame with
    columns ``day`` (a ``datetime64`` at midnight) and ``revenue`` (float),
    sorted ascending by day. Days with no trips are simply absent.
    """
    pickup = pd.to_datetime(df[PICKUP_COL])
    day = pickup.dt.normalize()
    revenue = df[FARE_COL].to_numpy() + df[TIP_COL].to_numpy()
    grouped = (
        pd.DataFrame({"day": day.to_numpy(), "revenue": revenue})
        .groupby("day", sort=True)["revenue"]
        .sum()
    )
    return pd.DataFrame(
        {
            "day": grouped.index.to_numpy(),
            "revenue": grouped.to_numpy().astype(float),
        }
    )


def anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag IQR-based outliers on fare and trip duration, per row.

    For each of ``fare_amount`` and ``trip_minutes`` the inter-quartile range
    ``IQR = Q3 - Q1`` is computed, and a value is an outlier when it falls below
    ``Q1 - 1.5 * IQR`` or above ``Q3 + 1.5 * IQR`` (the standard Tukey fence).
    When ``IQR == 0`` (all values identical, or fewer than the spread needed)
    the fence collapses to the single value and **no** row is flagged.

    Returns a frame aligned 1:1 with the (reset) input index, with boolean
    columns ``fare_outlier``, ``trip_minutes_outlier`` and ``is_outlier`` (the
    logical OR of the two). An empty input yields an empty, correctly-typed
    frame.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            {
                "fare_outlier": pd.Series([], dtype=bool),
                "trip_minutes_outlier": pd.Series([], dtype=bool),
                "is_outlier": pd.Series([], dtype=bool),
            }
        )

    def _fence(series: pd.Series) -> np.ndarray:
        values = series.to_numpy(dtype=float)
        q1 = np.quantile(values, 0.25)
        q3 = np.quantile(values, 0.75)
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros(n, dtype=bool)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (values < lower) | (values > upper)

    fare_out = _fence(df[FARE_COL])
    trip_out = _fence(df[TRIP_MINUTES_COL])
    return pd.DataFrame(
        {
            "fare_outlier": fare_out,
            "trip_minutes_outlier": trip_out,
            "is_outlier": fare_out | trip_out,
        }
    )
