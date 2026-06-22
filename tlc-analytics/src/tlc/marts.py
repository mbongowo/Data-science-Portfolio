"""Aggregated marts for cleaned TLC trips (pure pandas, no engine deps).

These four functions are the *reference* definitions of the analytics workload.
Each takes a cleaned frame (the output of :func:`tlc.clean.clean_trips`, i.e.
with ``trip_minutes`` and ``tip_pct`` present) and returns a small aggregated
DataFrame. The engine wrappers in ``engines.py`` express the identical logic in
SQL / Spark; these pandas versions are the source of truth and the basis of the
known-answer tests.

The aggregations:

* :func:`hourly_demand`       — trip counts by pickup hour (0-23).
* :func:`demand_by_dow`       — trip counts by day of week (0=Mon .. 6=Sun).
* :func:`tip_rate_by_payment` — mean ``tip_pct`` by payment type.
* :func:`fare_summary`        — count, mean/median fare and mean trip duration.
"""

from __future__ import annotations

import pandas as pd

PICKUP_COL = "pickup_datetime"
FARE_COL = "fare_amount"
PAYMENT_COL = "payment_type"


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
