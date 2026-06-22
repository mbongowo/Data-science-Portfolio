"""Cleaning rules for NYC TLC trip records (pure pandas, no engine deps).

The raw trip-record feed carries data-entry errors, disputed trips, and logging
artefacts: zero-fare rows, trips with no passengers, negative durations, and
fares in the thousands of dollars. :func:`clean_trips` applies a small set of
explicit, documented predicates and adds the two derived columns the marts need.

This is the pure-pandas reference path. The same logical filter is expressed in
SQL / Spark in the engine wrappers (``engines.py``), but the rules below are the
single source of truth and are covered by hand-derived known-answer tests.

Cleaning predicates (a row is **kept** only if all hold):

* ``fare_amount > 0``            — drop zero / negative fares.
* ``passenger_count > 0``        — drop trips that logged no passengers.
* ``dropoff > pickup``           — drop non-positive duration (pickup at or
  after dropoff).
* ``fare_amount <= fare_cap``    — drop implausibly large fares (default cap
  500.0; data-entry errors and disputes).

Derived columns added to the kept rows:

* ``trip_minutes`` — ``(dropoff - pickup)`` in minutes (float).
* ``tip_pct``      — ``tip_amount / fare_amount`` (fraction, not percent).
"""

from __future__ import annotations

import pandas as pd

#: Default upper bound on a plausible single-trip fare, in dollars.
DEFAULT_FARE_CAP = 500.0

#: Canonical column names the cleaner operates on.
PICKUP_COL = "pickup_datetime"
DROPOFF_COL = "dropoff_datetime"
FARE_COL = "fare_amount"
TIP_COL = "tip_amount"
PASSENGERS_COL = "passenger_count"


def clean_trips(
    df: pd.DataFrame, *, fare_cap: float = DEFAULT_FARE_CAP
) -> pd.DataFrame:
    """Drop invalid trip rows and add derived columns.

    Parameters
    ----------
    df:
        Trip records with at least the columns ``pickup_datetime``,
        ``dropoff_datetime``, ``fare_amount``, ``tip_amount`` and
        ``passenger_count``. The datetime columns may be strings or
        ``datetime64``; they are coerced with :func:`pandas.to_datetime`.
    fare_cap:
        Drop rows whose ``fare_amount`` exceeds this value. Defaults to
        :data:`DEFAULT_FARE_CAP`.

    Returns
    -------
    pandas.DataFrame
        A new frame containing only the rows that pass every predicate, with
        ``trip_minutes`` and ``tip_pct`` appended and the index reset.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    required = [PICKUP_COL, DROPOFF_COL, FARE_COL, TIP_COL, PASSENGERS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"clean_trips: missing required columns: {missing}")

    out = df.copy()
    out[PICKUP_COL] = pd.to_datetime(out[PICKUP_COL])
    out[DROPOFF_COL] = pd.to_datetime(out[DROPOFF_COL])

    duration = out[DROPOFF_COL] - out[PICKUP_COL]
    trip_minutes = duration.dt.total_seconds() / 60.0

    keep = (
        (out[FARE_COL] > 0)
        & (out[PASSENGERS_COL] > 0)
        & (trip_minutes > 0)
        & (out[FARE_COL] <= fare_cap)
    )

    out = out.loc[keep].copy()
    out["trip_minutes"] = trip_minutes.loc[keep].to_numpy()
    out["tip_pct"] = out[TIP_COL] / out[FARE_COL]
    return out.reset_index(drop=True)
