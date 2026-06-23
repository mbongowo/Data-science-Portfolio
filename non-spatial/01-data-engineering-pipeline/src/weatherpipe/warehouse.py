"""Warehouse helpers: load frames and run SQL against DuckDB.

This module is the boundary to the local warehouse. Every DuckDB call imports the
driver lazily, so importing this module is cheap and the test suite never pulls
in ``duckdb``. The same three primitives (load a frame, run a SQL file, run a
query) are all the local free path needs; the Azure path swaps the DuckDB
connection for an Azure SQL connection at the dbt profile, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataframe(
    df: pd.DataFrame,
    db_path: str | Path,
    table: str,
    *,
    schema: str = "raw",
) -> int:
    """Write ``df`` into ``schema.table`` in the DuckDB file, replacing it.

    Returns the row count written. Creates the schema and the parent directory if
    needed. ``duckdb`` is imported lazily.
    """
    import duckdb

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.register("incoming", df)
        con.execute(
            f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM incoming"
        )
        con.unregister("incoming")
    finally:
        con.close()
    return int(len(df))


def run_sql_file(db_path: str | Path, sql_path: str | Path) -> None:
    """Execute every statement in a ``.sql`` file against the DuckDB database."""
    import duckdb

    sql = Path(sql_path).read_text(encoding="utf-8")
    con = duckdb.connect(str(db_path))
    try:
        con.execute(sql)
    finally:
        con.close()


def query(db_path: str | Path, sql: str) -> pd.DataFrame:
    """Run ``sql`` against the DuckDB database and return the result as a frame."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()
