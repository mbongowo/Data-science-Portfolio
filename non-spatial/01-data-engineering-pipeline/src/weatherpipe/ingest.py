"""Ingestion: pull historical weather, parse it, and land it in the lake.

The data source is the free Open-Meteo historical (ERA5) API, which needs no API
key. :func:`fetch_open_meteo` is the only function that touches the network, and
it imports ``requests`` lazily so the rest of this module (and the test suite)
stays dependency-light.

The remaining functions are pure:

* :func:`parse_open_meteo` turns one API payload into a list of per-day record
  dicts.
* :func:`normalize` tidies a list of records for one station into a typed pandas
  DataFrame with the canonical :data:`WEATHER_COLUMNS`.
* :func:`partition_path` returns the ``station=/year=/month=`` lake path for a
  record, the single place that knows the lake layout.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

#: Canonical tidy schema, outermost identity columns first.
WEATHER_COLUMNS = [
    "station",
    "date",
    "tmin_c",
    "tmax_c",
    "tmean_c",
    "precip_mm",
]

# Open-Meteo daily variables we request, in the order their arrays line up.
_DAILY_VARS = (
    "temperature_2m_min",
    "temperature_2m_max",
    "temperature_2m_mean",
    "precipitation_sum",
)


def fetch_open_meteo(
    lat: float,
    lon: float,
    start: str,
    end: str,
    *,
    timeout: float = 60.0,
) -> dict:
    """Fetch daily historical weather for one point from Open-Meteo.

    Hits the free, no-key historical (archive) endpoint and returns the raw JSON
    payload as a dict. ``start`` / ``end`` are ISO dates (``YYYY-MM-DD``).
    ``requests`` is imported lazily so importing this module never requires it.
    """
    import requests  # lazy: not needed for parse/normalize or the tests

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(_DAILY_VARS),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_open_meteo(payload: dict) -> list[dict]:
    """Parse an Open-Meteo payload into a list of per-day record dicts (pure).

    Reads the ``daily`` block, zips the parallel arrays by index, and emits one
    dict per day with keys ``date``, ``tmin_c``, ``tmax_c``, ``tmean_c``,
    ``precip_mm``. Missing values (Open-Meteo sends JSON ``null``) pass through as
    ``None`` for the validator to handle. An absent ``daily`` block yields ``[]``.
    """
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    tmin = daily.get("temperature_2m_min") or []
    tmax = daily.get("temperature_2m_max") or []
    tmean = daily.get("temperature_2m_mean") or []
    precip = daily.get("precipitation_sum") or []

    records: list[dict] = []
    for i, date in enumerate(dates):
        records.append(
            {
                "date": date,
                "tmin_c": _at(tmin, i),
                "tmax_c": _at(tmax, i),
                "tmean_c": _at(tmean, i),
                "precip_mm": _at(precip, i),
            }
        )
    return records


def normalize(records: Iterable[dict], station: str) -> pd.DataFrame:
    """Tidy per-day records for one ``station`` into a typed DataFrame (pure).

    Returns a DataFrame with exactly :data:`WEATHER_COLUMNS`: a ``station`` label,
    a ``date`` cast to ``datetime64``, and the four float measures. An empty input
    yields an empty, correctly-typed frame so downstream code can rely on the
    schema.
    """
    rows = list(records)
    if not rows:
        empty = pd.DataFrame({c: [] for c in WEATHER_COLUMNS})
        empty["date"] = pd.to_datetime(empty["date"])
        for col in ("tmin_c", "tmax_c", "tmean_c", "precip_mm"):
            empty[col] = empty[col].astype("float64")
        empty["station"] = empty["station"].astype("object")
        return empty[WEATHER_COLUMNS]

    df = pd.DataFrame(rows)
    df.insert(0, "station", station)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("tmin_c", "tmax_c", "tmean_c", "precip_mm"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df[WEATHER_COLUMNS]


def partition_path(station: str, date) -> str:
    """Return the ``station=<s>/year=<Y>/month=<M>`` lake path for a record (pure).

    ``date`` is anything :func:`pandas.Timestamp` accepts (an ISO string or a
    datetime). ``month`` is zero-padded to two digits (Hive convention); ``year``
    is not padded. The station label is slugified (lower-cased, spaces and
    accents-free separators to underscores) so it is filesystem-safe. The
    separator is always ``/`` so the value is stable across platforms.
    """
    ts = pd.Timestamp(date)
    slug = _slug(station)
    return f"station={slug}/year={ts.year}/month={ts.month:02d}"


def _slug(station: str) -> str:
    """Lower-case, ASCII-fold and underscore-separate a station name."""
    import unicodedata

    folded = (
        unicodedata.normalize("NFKD", str(station))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    out = []
    for ch in folded.lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _at(arr: list, i: int):
    """Safe positional get: return ``None`` if ``arr`` is short."""
    return arr[i] if i < len(arr) else None
