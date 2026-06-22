"""tlc-analytics: NYC TLC trip-record analytics at billions-of-rows scale.

A tabular engine bake-off. The same aggregation workload — demand by hour and
day of week, tipping by payment type, fare and duration summaries — is run over
multi-year, Hive-partitioned TLC Parquet through different engines (Spark,
DuckDB, a managed warehouse) and the engines are ranked on runtime, memory, and
cost. No geospatial columns are used.

The package is split so that the interpretation-critical numeric core (cleaning
rules and the aggregated marts, in pure pandas) has no engine dependency and is
always importable and testable. The engine runners that need ``duckdb`` /
``pyspark`` live in :mod:`tlc.engines` and are imported lazily, never by this
package's ``__init__`` or by the test suite.
"""

from __future__ import annotations

from tlc.benchmark import (
    BenchmarkResult,
    bake_off,
    duckdb_available,
    run_duckdb_query,
    summarize,
    time_callable,
)
from tlc.clean import clean_trips
from tlc.marts import (
    anomaly_flags,
    demand_by_dow,
    fare_summary,
    hourly_demand,
    revenue_by_day,
    tip_rate_by_payment,
    trip_duration_buckets,
)
from tlc.partitions import iter_partitions, partition_relpath

__all__ = [
    "clean_trips",
    "hourly_demand",
    "demand_by_dow",
    "tip_rate_by_payment",
    "fare_summary",
    "trip_duration_buckets",
    "revenue_by_day",
    "anomaly_flags",
    "iter_partitions",
    "partition_relpath",
    "time_callable",
    "summarize",
    "BenchmarkResult",
    "bake_off",
    "run_duckdb_query",
    "duckdb_available",
    "__version__",
]

__version__ = "0.1.0"
