"""dwh: the pure-Python core of a modern data stack worked example.

This package backs a worked "modern data stack" project: batch ETL loads raw
IMDb extracts into a DuckDB warehouse, layered dbt models (staging ->
intermediate -> marts) transform and test them, an orchestrator schedules the
run, and a BI layer reads the marts.

The importable, always-tested core here has **no dbt / duckdb / orchestrator
dependency**:

* :mod:`dwh.dq` — a pure-pandas data-quality runner that mirrors dbt's four
  generic tests (not_null / unique / accepted_values / relationships).
* :mod:`dwh.dimensional` — surrogate keys and a date dimension, the same
  building blocks the dbt marts express in SQL.

The heavy pieces (warehouse loading, dbt invocation, Airflow/Dagster DAGs) live
in :mod:`dwh.orchestration` and :mod:`dwh.cli`, which import their dependencies
lazily and are not imported by this module or by the test suite.
"""

from __future__ import annotations

from dwh.dimensional import (
    SCD2_FAR_FUTURE,
    build_date_dim,
    scd2_snapshot,
    surrogate_key,
)
from dwh.dq import (
    TestResult,
    TestSpec,
    run_suite,
    suite_failed,
    test_accepted_range,
    test_accepted_values,
    test_expression,
    test_freshness,
    test_not_null,
    test_not_null_where,
    test_relationships,
    test_unique,
)

__all__ = [
    "test_not_null",
    "test_unique",
    "test_accepted_values",
    "test_relationships",
    "test_accepted_range",
    "test_not_null_where",
    "test_expression",
    "test_freshness",
    "run_suite",
    "suite_failed",
    "TestResult",
    "TestSpec",
    "surrogate_key",
    "build_date_dim",
    "scd2_snapshot",
    "SCD2_FAR_FUTURE",
    "__version__",
]

__version__ = "0.1.0"
