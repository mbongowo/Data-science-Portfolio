"""Ingestion: pull air-quality readings from the Open-Meteo Air Quality API.

The data source is the free Open-Meteo Air Quality API, which needs no API key.
:func:`fetch_open_meteo_aq` is the only function that touches the network, and it
imports ``requests`` lazily so the rest of this module (and the test suite) stays
dependency-light. :func:`parse_aq` is pure and turns one payload into a list of
reading dicts on the canonical ``(ts, station, pm25, pm10)`` schema the windowing
and alert core consume.
"""

from __future__ import annotations

from datetime import datetime, timezone

#: Canonical reading schema produced by the ingest layer.
READING_COLUMNS = ["ts", "station", "pm25", "pm10"]

# Open-Meteo hourly variables we request, in the order their arrays line up.
_HOURLY_VARS = ("pm2_5", "pm10")


def fetch_open_meteo_aq(
    lat: float,
    lon: float,
    hours: int = 72,
    *,
    timeout: float = 60.0,
) -> dict:
    """Fetch recent hourly PM2.5 / PM10 for one point from Open-Meteo.

    Hits the free, no-key Air Quality endpoint and returns the raw JSON payload
    as a dict, covering roughly the past ``hours`` hours. ``requests`` is
    imported lazily so importing this module never requires it.
    """
    import requests  # lazy: not needed for parse or the tests

    past_days = max(1, (hours + 23) // 24)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_VARS),
        "past_days": past_days,
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_aq(payload: dict, station: str) -> list[dict]:
    """Parse an Open-Meteo air-quality payload into reading dicts (pure).

    Reads the ``hourly`` block, zips the parallel arrays by index, and emits one
    dict per hour with keys ``ts`` (epoch seconds, int), ``station``, ``pm25``,
    ``pm10``. Hours where *both* pollutants are missing are skipped; a single
    missing pollutant passes through as ``None``. An absent ``hourly`` block
    yields ``[]``.
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    pm25 = hourly.get("pm2_5") or []
    pm10 = hourly.get("pm10") or []

    readings: list[dict] = []
    for i, t in enumerate(times):
        v25 = _at(pm25, i)
        v10 = _at(pm10, i)
        if v25 is None and v10 is None:
            continue
        readings.append(
            {
                "ts": _to_epoch(t),
                "station": station,
                "pm25": v25,
                "pm10": v10,
            }
        )
    return readings


def _to_epoch(t: str) -> int:
    """Convert an Open-Meteo ISO time string (UTC) to epoch seconds."""
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _at(arr: list, i: int):
    """Safe positional get: return ``None`` if ``arr`` is short."""
    return arr[i] if i < len(arr) else None
