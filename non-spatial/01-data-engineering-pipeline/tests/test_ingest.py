"""Known-answer tests for ingestion: parse, normalize, partition path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weatherpipe.ingest import (
    WEATHER_COLUMNS,
    normalize,
    parse_open_meteo,
    partition_path,
)

# A tiny, hand-written Open-Meteo archive payload: two days for one point.
_PAYLOAD = {
    "latitude": 4.05,
    "longitude": 9.7,
    "daily": {
        "time": ["2023-01-01", "2023-01-02"],
        "temperature_2m_min": [22.1, 23.0],
        "temperature_2m_max": [31.4, 30.2],
        "temperature_2m_mean": [26.5, 26.0],
        "precipitation_sum": [0.0, 5.4],
    },
}


def test_parse_open_meteo_zips_arrays() -> None:
    records = parse_open_meteo(_PAYLOAD)
    assert records == [
        {
            "date": "2023-01-01",
            "tmin_c": 22.1,
            "tmax_c": 31.4,
            "tmean_c": 26.5,
            "precip_mm": 0.0,
        },
        {
            "date": "2023-01-02",
            "tmin_c": 23.0,
            "tmax_c": 30.2,
            "tmean_c": 26.0,
            "precip_mm": 5.4,
        },
    ]


def test_parse_open_meteo_missing_daily_is_empty() -> None:
    assert parse_open_meteo({}) == []
    assert parse_open_meteo({"daily": {}}) == []


def test_normalize_columns_and_types() -> None:
    records = parse_open_meteo(_PAYLOAD)
    df = normalize(records, "Douala")

    assert list(df.columns) == WEATHER_COLUMNS
    assert len(df) == 2
    assert (df["station"] == "Douala").all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    for col in ("tmin_c", "tmax_c", "tmean_c", "precip_mm"):
        assert df[col].dtype == np.float64
    assert df.loc[0, "date"] == pd.Timestamp("2023-01-01")
    assert df.loc[1, "precip_mm"] == 5.4


def test_normalize_empty_keeps_schema() -> None:
    df = normalize([], "Maroua")
    assert list(df.columns) == WEATHER_COLUMNS
    assert len(df) == 0


def test_partition_path_exact_string() -> None:
    assert (
        partition_path("Douala", "2023-01-05")
        == "station=douala/year=2023/month=01"
    )
    # Accented / spaced names slugify; month zero-pads.
    assert (
        partition_path("Yaoundé", pd.Timestamp("2023-12-31"))
        == "station=yaounde/year=2023/month=12"
    )
