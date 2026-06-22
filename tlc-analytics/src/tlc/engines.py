"""Engine runners for the bake-off (lazy heavy imports).

These wrappers execute the aggregation workload against a real engine over the
partitioned Parquet lake. They are deliberately **not** imported by the package
``__init__`` or the test suite: ``duckdb`` and ``pyspark`` are imported lazily
inside each function so that the pure-pandas core, and CI, run without the
engine stack installed.

The SQL / Spark logic here mirrors the pandas reference marts in ``marts.py``;
those pandas definitions are the source of truth.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def run_duckdb(sql: str, parquet_glob: str) -> pd.DataFrame:
    """Run a SQL query against the Parquet lake with DuckDB, in-process.

    Parameters
    ----------
    sql:
        A SQL statement. Reference the lake with the ``{glob}`` placeholder,
        which is substituted with a ``read_parquet`` over ``parquet_glob``.
    parquet_glob:
        Glob expanded by DuckDB's Parquet reader (Hive partitions discovered
        automatically with ``hive_partitioning = true``).

    Returns
    -------
    pandas.DataFrame
        The query result.
    """
    import duckdb  # lazy: engine-only dependency

    relation = f"read_parquet('{parquet_glob}', hive_partitioning = true)"
    statement = sql.format(glob=relation)
    con = duckdb.connect()
    try:
        return con.execute(statement).fetch_df()
    finally:
        con.close()


def run_spark(
    sql: str,
    parquet_glob: str,
    *,
    app_name: str = "tlc-analytics",
    spark: Any = None,
) -> pd.DataFrame:
    """Run a SQL query against the Parquet lake with Spark.

    Parameters
    ----------
    sql:
        A SQL statement referencing the temp view ``trips``.
    parquet_glob:
        Path / glob Spark reads as Parquet (partition columns are inferred).
    app_name:
        Spark application name, used only when this function creates the session.
    spark:
        An existing ``SparkSession`` to reuse (e.g. on a cluster). If ``None``,
        a local session is created and torn down here.

    Returns
    -------
    pandas.DataFrame
        The query result, collected to the driver via ``toPandas``.
    """
    from pyspark.sql import SparkSession  # lazy: engine-only dependency

    own_session = spark is None
    if own_session:
        spark = SparkSession.builder.appName(app_name).getOrCreate()

    try:
        frame = spark.read.parquet(parquet_glob)
        frame.createOrReplaceTempView("trips")
        return spark.sql(sql).toPandas()
    finally:
        if own_session:
            spark.stop()
