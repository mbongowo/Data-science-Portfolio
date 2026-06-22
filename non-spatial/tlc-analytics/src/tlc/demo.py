"""A one-command, reproducible demo of the pandas reference core.

The engine bake-off (Spark vs DuckDB vs warehouse, over billions of partitioned
Parquet rows) cannot run in a seconds-long, dependency-free demo. What *can* run
anywhere — including CI, with only ``numpy`` / ``pandas`` / ``pyyaml`` + stdlib —
is the **pure-pandas numeric core**: the cleaning rules and the four aggregated
marts that are the source of truth for the whole project.

:func:`run_demo` deterministically synthesises a small NYC-taxi-like trips frame
(seeded with :func:`numpy.random.default_rng`), drives the *real* core
(:func:`tlc.clean.clean_trips` then the marts), times the pandas mart build with
the *real* benchmark harness (:func:`tlc.time_callable`), writes honest artefacts
to ``out_dir`` and returns the headline numbers. Runtime is well under a second.

The synthetic frame has realistic structure so the marts produce meaningful
insights rather than noise:

* **Rush-hour demand peaks** — pickups are sampled from an hour-of-day weight
  curve with a morning (~8h) and evening (~18h) commute hump.
* **Card tips beat cash tips** — card riders tip a positive fraction of fare;
  cash tips are essentially unrecorded by the meter (the well-known TLC
  artefact), so mean ``tip_pct`` for cash is ~0.
* **Planted bad rows** — a fixed handful of rows violate the cleaning predicates
  (zero/negative fare, zero passengers, non-positive duration, over the fare
  cap) so the demo demonstrates the cleaner dropping exactly those rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tlc import marts
from tlc.benchmark import (
    BenchmarkResult,
    bake_off,
    duckdb_available,
    summarize,
    time_callable,
)
from tlc.clean import clean_trips

#: Number of synthetic "good" trips before the planted bad rows are appended.
N_GOOD_TRIPS = 5_000

#: Relative pickup demand by hour of day (0-23): morning + evening commute humps.
_HOUR_WEIGHTS = np.array(
    [
        0.3, 0.2, 0.15, 0.1, 0.15, 0.4, 1.0, 2.2,  # 0-7
        3.0, 2.4, 1.6, 1.5, 1.7, 1.6, 1.5, 1.7,    # 8-15
        2.2, 2.9, 3.2, 2.6, 1.9, 1.5, 1.0, 0.6,    # 16-23
    ],
    dtype=float,
)

#: Number of planted bad rows (one per cleaning predicate, plus a duplicate).
N_BAD_ROWS = 5


def synthesize_trips(seed: int = 0) -> pd.DataFrame:
    """Deterministically build a small NYC-taxi-like trips frame.

    Parameters
    ----------
    seed:
        Seed for :func:`numpy.random.default_rng`. The same seed always yields
        the identical frame.

    Returns
    -------
    pandas.DataFrame
        ``N_GOOD_TRIPS`` realistic rows followed by :data:`N_BAD_ROWS` planted
        invalid rows, with the raw columns the cleaner expects plus derived
        ``hour`` / ``dow`` convenience columns.
    """
    rng = np.random.default_rng(seed)
    n = N_GOOD_TRIPS

    # Spread pickups across a two-week January window, weighting the hour of day
    # by the commute curve so hourly_demand shows real peaks.
    day_offset = rng.integers(0, 14, size=n)
    weights = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()
    hour = rng.choice(24, size=n, p=weights)
    minute = rng.integers(0, 60, size=n)

    base = np.datetime64("2023-01-02T00:00:00")  # a Monday
    pickup = (
        base
        + day_offset.astype("timedelta64[D]")
        + hour.astype("timedelta64[h]")
        + minute.astype("timedelta64[m]")
    )

    trip_minutes = rng.gamma(shape=2.0, scale=6.0, size=n) + 1.0
    dropoff = pickup + (trip_minutes * 60).astype("timedelta64[s]")

    passenger_count = rng.integers(1, 5, size=n)
    # Fare loosely tracks duration plus a base and noise.
    fare_amount = np.round(
        3.0 + 1.8 * trip_minutes + rng.normal(0.0, 3.0, size=n), 2
    )
    fare_amount = np.clip(fare_amount, 3.0, None)

    # ~60% card, ~40% cash. Card riders tip a positive fraction; cash tips are
    # essentially never recorded by the meter (the real TLC logging artefact).
    payment_type = rng.choice(["card", "cash"], size=n, p=[0.6, 0.4])
    tip_frac = np.where(
        payment_type == "card",
        rng.normal(0.18, 0.04, size=n).clip(0.0, None),
        0.0,
    )
    tip_amount = np.round(tip_frac * fare_amount, 2)

    good = pd.DataFrame(
        {
            "pickup_datetime": pickup,
            "dropoff_datetime": dropoff,
            "passenger_count": passenger_count.astype(int),
            "fare_amount": fare_amount,
            "tip_amount": tip_amount,
            "payment_type": payment_type,
        }
    )

    # Five planted bad rows, one per cleaning predicate (plus an over-cap fare),
    # all of which clean_trips must drop.
    p0 = good["pickup_datetime"].iloc[0]
    bad = pd.DataFrame(
        {
            "pickup_datetime": [p0, p0, p0, p0, p0],
            "dropoff_datetime": [
                p0 + np.timedelta64(10, "m"),  # zero fare
                p0 + np.timedelta64(10, "m"),  # zero passengers
                p0 - np.timedelta64(5, "m"),   # dropoff before pickup
                p0 + np.timedelta64(10, "m"),  # over fare cap
                p0,                            # zero duration (pickup == dropoff)
            ],
            "passenger_count": [1, 0, 1, 1, 1],
            "fare_amount": [0.0, 12.0, 12.0, 999.0, 12.0],
            "tip_amount": [0.0, 1.0, 1.0, 1.0, 1.0],
            "payment_type": ["card", "card", "card", "card", "card"],
        }
    )

    out = pd.concat([good, bad], ignore_index=True)
    out["hour"] = out["pickup_datetime"].dt.hour
    out["dow"] = out["pickup_datetime"].dt.dayofweek
    return out


def run_demo(seed: int = 0, out_dir: str = "outputs") -> dict[str, Any]:
    """Synthesize trips, drive the real core, time it, and write artefacts.

    Parameters
    ----------
    seed:
        Seed passed to :func:`synthesize_trips`; controls the whole run.
    out_dir:
        Directory for the CSV/JSON artefacts (created if absent).

    Returns
    -------
    dict
        ``rows_in``, ``rows_after_clean``, ``peak_demand_hour``,
        ``tip_rate_card``, ``tip_rate_cash``, ``mean_fare`` and
        ``pandas_mart_build_seconds`` (from the real benchmark harness).
    """
    raw = synthesize_trips(seed)
    rows_in = int(len(raw))

    trips = clean_trips(raw)
    rows_after_clean = int(len(trips))

    # Time the real pandas mart build with the real harness. The timed callable
    # builds all four marts so the timing reflects the whole reference workload.
    def _build_all() -> dict[str, pd.DataFrame]:
        return {
            "hourly_demand": marts.hourly_demand(trips),
            "demand_by_dow": marts.demand_by_dow(trips),
            "tip_rate_by_payment": marts.tip_rate_by_payment(trips),
            "fare_summary": marts.fare_summary(trips),
            "trip_duration_buckets": marts.trip_duration_buckets(trips),
            "revenue_by_day": marts.revenue_by_day(trips),
            "anomaly_flags": marts.anomaly_flags(trips),
        }

    built, seconds = time_callable(_build_all)
    ranking = summarize(
        [BenchmarkResult(engine="pandas", query="marts", seconds=seconds)]
    )

    # Real engine bake-off on the demo data: pandas vs DuckDB when DuckDB is
    # importable, otherwise a clean "engine unavailable" skip row. This is a
    # *measured* comparison on this frame, not the projected billions-row table.
    engine_bakeoff = bake_off(trips, query="hourly_demand")

    hd = built["hourly_demand"]
    peak_demand_hour = int(hd.loc[hd["trips"].idxmax(), "hour"])

    tip = built["tip_rate_by_payment"]
    tip_map = dict(zip(tip["payment_type"], tip["mean_tip_pct"], strict=True))
    tip_rate_card = float(tip_map.get("card", float("nan")))
    tip_rate_cash = float(tip_map.get("cash", float("nan")))

    mean_fare = float(built["fare_summary"].iloc[0]["mean_fare"])

    # The busiest single revenue day, straight off the revenue_by_day mart.
    rev = built["revenue_by_day"]
    peak_idx = int(rev["revenue"].idxmax())
    peak_revenue_day = str(pd.Timestamp(rev.loc[peak_idx, "day"]).date())
    peak_revenue = float(rev.loc[peak_idx, "revenue"])

    # IQR anomaly count on the cleaned frame (fare or duration outliers).
    n_anomalies = int(built["anomaly_flags"]["is_outlier"].sum())

    result = {
        "rows_in": rows_in,
        "rows_after_clean": rows_after_clean,
        "peak_demand_hour": peak_demand_hour,
        "tip_rate_card": tip_rate_card,
        "tip_rate_cash": tip_rate_cash,
        "mean_fare": mean_fare,
        "peak_revenue_day": peak_revenue_day,
        "peak_revenue": peak_revenue,
        "n_anomalies": n_anomalies,
        "pandas_mart_build_seconds": float(seconds),
        "duckdb_available": bool(duckdb_available()),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    built["hourly_demand"].to_csv(out_path / "hourly_demand.csv", index=False)
    built["tip_rate_by_payment"].to_csv(
        out_path / "tip_rate_by_payment.csv", index=False
    )
    built["fare_summary"].to_csv(out_path / "fare_summary.csv", index=False)
    built["trip_duration_buckets"].to_csv(
        out_path / "trip_duration_buckets.csv", index=False
    )
    built["revenue_by_day"].to_csv(out_path / "revenue_by_day.csv", index=False)
    ranking.to_csv(out_path / "benchmark.csv", index=False)
    engine_bakeoff.to_csv(out_path / "engine_bakeoff.csv", index=False)
    with open(out_path / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result
