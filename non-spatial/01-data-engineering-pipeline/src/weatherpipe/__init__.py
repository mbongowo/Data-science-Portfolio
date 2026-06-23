"""weatherpipe: the pure-Python core of a Cameroon weather data pipeline.

This package backs an end-to-end data-engineering project shaped like the
Data-Engineering-Zoomcamp capstone: historical weather for five Cameroon cities
is **ingested** from the free Open-Meteo API, landed in a partitioned lake, loaded
into a warehouse, transformed by **dbt** marts with tests, run on a schedule, and
read by a dashboard. There are two deployment paths: a free local one on DuckDB
and an opt-in Azure one provisioned with Terraform.

The importable, always-tested core here needs only numpy / pandas / pyyaml +
stdlib — no requests, duckdb, dbt, prefect, azure or streamlit:

* :mod:`weatherpipe.ingest`    — parse / normalize Open-Meteo records and the
  ``station=/year=/month=`` lake layout (the ``requests`` call is lazy).
* :mod:`weatherpipe.validate`  — range / null / duplicate checks returning a
  cleaned frame and a rejection report.
* :mod:`weatherpipe.transform` — pure-pandas marts mirroring the dbt models.

The heavy pieces (warehouse loading, the Prefect flow, the dbt build, the
Streamlit dashboard, the Azure deployment) live in :mod:`weatherpipe.warehouse`,
:mod:`weatherpipe.orchestrate`, :mod:`weatherpipe.cli` and the ``transform`` /
``terraform`` / ``app`` directories, which import their dependencies lazily and
are not imported by this module or by the test suite.
"""

from __future__ import annotations

from weatherpipe.ingest import (
    WEATHER_COLUMNS,
    normalize,
    parse_open_meteo,
    partition_path,
)
from weatherpipe.transform import daily_summary, monthly_summary
from weatherpipe.validate import validate_weather

__all__ = [
    "WEATHER_COLUMNS",
    "parse_open_meteo",
    "normalize",
    "partition_path",
    "validate_weather",
    "daily_summary",
    "monthly_summary",
    "__version__",
]

__version__ = "0.1.0"
