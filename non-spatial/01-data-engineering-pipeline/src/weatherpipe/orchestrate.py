"""Optional orchestration: a Prefect flow wiring the pipeline together.

This is the **optional** scheduling layer. The free path runs end to end without
it (``weatherpipe demo``, then ``dbt build``, then the dashboard); this module
adds a Prefect flow that runs the same steps on a schedule for a real deployment.

Prefect is imported lazily inside :func:`weather_flow`, so importing this module
costs nothing and the test suite never pulls in Prefect. The flow shape mirrors
the project:

    ingest  ->  validate  ->  load (warehouse)  ->  dbt build (marts + tests)

Each step is a thin wrapper around the pure core (:mod:`weatherpipe.ingest`,
:mod:`weatherpipe.validate`) and the lazy edges (:mod:`weatherpipe.warehouse`,
the dbt CLI), so the scheduled run and the local demo exercise the same logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the pipeline YAML config."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def weather_flow(config_path: str | Path = "config/config.yaml") -> dict[str, int]:
    """Run the full pipeline as a Prefect flow: ingest -> validate -> load -> dbt.

    Returns a small dict of per-station clean-row counts loaded into the
    warehouse. Prefect (and the lazy edges it calls) are imported inside, so this
    function is the only place that needs them.
    """
    import pandas as pd
    from prefect import flow, task

    from weatherpipe.ingest import fetch_open_meteo, normalize, parse_open_meteo
    from weatherpipe.validate import validate_weather
    from weatherpipe.warehouse import load_dataframe

    cfg = load_config(config_path)
    stations = cfg["stations"]
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    db_path = cfg["warehouse"]["duckdb_path"]

    @task(retries=2, retry_delay_seconds=10)
    def ingest_one(station: dict) -> pd.DataFrame:
        payload = fetch_open_meteo(station["lat"], station["lon"], start, end)
        return normalize(parse_open_meteo(payload), station["name"])

    @task
    def validate_all(frames: list[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(frames, ignore_index=True)
        clean, report = validate_weather(combined)
        print(f"validation: {report['n_clean']}/{report['n_input']} rows kept")
        return clean

    @task
    def load(clean: pd.DataFrame) -> int:
        return load_dataframe(clean, db_path, "weather", schema="raw")

    @task
    def dbt_build() -> int:
        from weatherpipe.cli import run_dbt_build

        return run_dbt_build()

    @flow(name="cameroon-weather")
    def _flow() -> dict[str, int]:
        frames = [ingest_one(s) for s in stations]
        clean = validate_all(frames)
        loaded = load(clean)
        dbt_build()
        return {"rows_loaded": loaded}

    return _flow()
