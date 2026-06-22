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

from tlc.benchmark import BenchmarkResult, summarize, time_callable
from tlc.clean import clean_trips
from tlc.marts import (
    demand_by_dow,
    fare_summary,
    hourly_demand,
    tip_rate_by_payment,
)

__all__ = [
    "clean_trips",
    "hourly_demand",
    "demand_by_dow",
    "tip_rate_by_payment",
    "fare_summary",
    "time_callable",
    "summarize",
    "BenchmarkResult",
    "__version__",
]

__version__ = "0.1.0"
