"""Storage sink: persist readings and alerts to DuckDB (guarded).

This module runs in the docker-compose stack, not in CI. ``duckdb`` is imported
lazily inside each function, so importing this module is free and the test suite
never needs DuckDB. The same readings/alerts that the pure core produces are what
get stored here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_READINGS_DDL = """
CREATE TABLE IF NOT EXISTS readings (
    ts BIGINT,
    station VARCHAR,
    pm25 DOUBLE,
    pm10 DOUBLE,
    aqi INTEGER,
    category VARCHAR
)
"""

_ALERTS_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    ts BIGINT,
    station VARCHAR,
    rule VARCHAR,
    value DOUBLE,
    severity INTEGER,
    message VARCHAR
)
"""


def _connect(db_path: str) -> Any:
    """Open (or create) a DuckDB database (lazy ``duckdb`` import)."""
    import duckdb

    con = duckdb.connect(db_path)
    con.execute(_READINGS_DDL)
    con.execute(_ALERTS_DDL)
    return con


def store_readings(readings: Iterable[dict], db_path: str = "aqstream.duckdb") -> int:
    """Insert reading dicts into the ``readings`` table. Returns the row count."""
    con = _connect(db_path)
    rows = [
        (
            int(r["ts"]),
            str(r["station"]),
            r.get("pm25"),
            r.get("pm10"),
            r.get("aqi"),
            r.get("category"),
        )
        for r in readings
    ]
    con.executemany(
        "INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    con.close()
    return len(rows)


def store_alerts(alerts: Iterable[Any], db_path: str = "aqstream.duckdb") -> int:
    """Insert :class:`aqstream.alerts.Alert` objects into ``alerts``. Returns count."""
    con = _connect(db_path)
    rows = [
        (
            int(a.ts),
            str(a.station),
            str(a.rule),
            float(a.value),
            int(a.severity),
            str(a.message),
        )
        for a in alerts
    ]
    con.executemany(
        "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    con.close()
    return len(rows)
