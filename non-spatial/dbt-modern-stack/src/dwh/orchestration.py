"""Orchestration: load raw data, invoke dbt, and build scheduler DAGs.

This module is the boundary between the pure-Python core and the heavy parts of
the stack (DuckDB, dbt, Airflow/Dagster). Every heavy import happens *inside* a
function, so importing this module is cheap and never drags in the warehouse or
a scheduler. The test suite does not import it.

Pipeline shape:

    seed_warehouse()   raw IMDb TSVs   -> DuckDB raw schema
    run_dbt_build()    DuckDB raw      -> staging -> intermediate -> marts (+ tests)
    build_*_dag()      wrap the above on a schedule with a source freshness gate

The freshness gate fails fast when the raw extracts are older than the
threshold in ``config/warehouse.yaml`` (IMDb publishes daily), so a stale load
does not silently propagate into the marts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the warehouse YAML config."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def seed_warehouse(config_path: str | Path) -> dict[str, int]:
    """Load the raw IMDb extracts into the DuckDB warehouse.

    Reads the gzipped TSV extracts named in ``config/warehouse.yaml`` and writes
    one raw table per source into the configured DuckDB file. Returns a mapping
    of table name -> row count loaded.

    DuckDB and pandas are imported lazily so this stays out of the core import
    path.
    """
    import duckdb
    import pandas as pd

    cfg = load_config(config_path)
    db_path = cfg["warehouse"]["duckdb_path"]
    raw_schema = cfg["warehouse"].get("raw_schema", "raw")
    sources = cfg["sources"]

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    loaded: dict[str, int] = {}
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {raw_schema}")
        for name, spec in sources.items():
            path = spec["path"]
            # IMDb extracts are gzipped, tab-separated, with "\N" for NULL.
            df = pd.read_csv(
                path,
                sep="\t",
                na_values="\\N",
                dtype=str,
                quoting=3,  # csv.QUOTE_NONE — IMDb fields are not quoted
            )
            con.register("incoming", df)
            con.execute(
                f"CREATE OR REPLACE TABLE {raw_schema}.{name} AS SELECT * FROM incoming"
            )
            con.unregister("incoming")
            loaded[name] = int(len(df))
    finally:
        con.close()
    return loaded


def run_dbt_build(
    project_dir: str | Path = "transform",
    profiles_dir: str | Path = "transform",
    *,
    select: str | None = None,
) -> int:
    """Invoke ``dbt build`` (run models + run tests) via the dbt programmatic API.

    Returns the process-style exit code (0 on success). ``dbt`` is imported
    lazily. ``select`` maps to dbt's ``--select`` for partial builds.
    """
    from dbt.cli.main import dbtRunner  # lazy import

    args = [
        "build",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(profiles_dir),
    ]
    if select:
        args += ["--select", select]

    res = dbtRunner().invoke(args)
    return 0 if res.success else 1


def build_airflow_dag(config_path: str | Path) -> Any:
    """Construct an Airflow DAG: freshness check -> seed -> dbt build.

    Imports Airflow lazily and returns the assembled ``DAG`` object so this can
    live in an Airflow ``dags/`` folder without importing Airflow at module load
    elsewhere. Failure handling: the freshness task fails the run if the raw
    extracts are stale, so the seed and build downstream never run on stale data.
    """
    from datetime import datetime, timedelta

    from airflow import DAG
    from airflow.operators.python import PythonOperator

    cfg = load_config(config_path)
    schedule = cfg["schedule"]["cron"]

    default_args = {
        "owner": "analytics",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id="imdb_dwh",
        schedule=schedule,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["dbt", "imdb", "modern-data-stack"],
    )

    check_freshness = PythonOperator(
        task_id="check_source_freshness",
        python_callable=assert_sources_fresh,
        op_args=[config_path],
        dag=dag,
    )
    seed = PythonOperator(
        task_id="seed_warehouse",
        python_callable=seed_warehouse,
        op_args=[config_path],
        dag=dag,
    )
    build = PythonOperator(
        task_id="dbt_build",
        python_callable=run_dbt_build,
        dag=dag,
    )

    check_freshness >> seed >> build
    return dag


def build_dagster_job(config_path: str | Path) -> Any:
    """Construct an equivalent Dagster job (freshness -> seed -> dbt build).

    Imports Dagster lazily. Same ordering and failure semantics as the Airflow
    variant; pick whichever scheduler your platform runs.
    """
    from dagster import job, op

    @op
    def check_freshness_op() -> None:
        assert_sources_fresh(config_path)

    @op
    def seed_op(_after: None) -> dict[str, int]:
        return seed_warehouse(config_path)

    @op
    def build_op(_loaded: dict[str, int]) -> int:
        return run_dbt_build()

    @job(name="imdb_dwh")
    def imdb_dwh_job() -> None:
        build_op(seed_op(check_freshness_op()))

    return imdb_dwh_job


def assert_sources_fresh(config_path: str | Path) -> None:
    """Raise if any raw source file is older than its freshness threshold.

    Mirrors dbt's source freshness check at the orchestration layer so the DAG
    fails *before* loading stale data. Thresholds come from
    ``config/warehouse.yaml``.
    """
    import time

    cfg = load_config(config_path)
    default_max = cfg.get("freshness", {}).get("max_age_hours", 48)
    now = time.time()

    stale: list[str] = []
    for name, spec in cfg["sources"].items():
        path = Path(spec["path"])
        max_age = spec.get("max_age_hours", default_max)
        if not path.exists():
            stale.append(f"{name}: missing ({path})")
            continue
        age_hours = (now - path.stat().st_mtime) / 3600.0
        if age_hours > max_age:
            stale.append(f"{name}: {age_hours:.1f}h old (> {max_age}h)")

    if stale:
        raise RuntimeError("Source freshness check failed:\n  " + "\n  ".join(stale))
