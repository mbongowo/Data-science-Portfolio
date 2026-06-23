"""Known-answer tests for the weather data-quality validator."""

from __future__ import annotations

import pandas as pd

from weatherpipe.validate import validate_weather


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _good(station: str, date: str, tmin, tmax, tmean, precip) -> dict:
    return {
        "station": station,
        "date": date,
        "tmin_c": tmin,
        "tmax_c": tmax,
        "tmean_c": tmean,
        "precip_mm": precip,
    }


def test_validate_drops_exactly_the_planted_bad_rows() -> None:
    rows = [
        _good("Douala", "2023-01-01", 22.0, 31.0, 26.5, 0.0),   # clean
        _good("Douala", "2023-01-02", 23.0, 30.0, 26.0, 5.4),   # clean
        _good("Maroua", "2023-01-01", 40.0, 10.0, 25.0, 1.0),   # tmin > tmax
        _good("Maroua", "2023-01-02", 20.0, 33.0, 27.0, -3.0),  # negative precip
        _good("Douala", "2023-01-01", 22.0, 31.0, 26.5, 0.0),   # duplicate key
    ]
    clean, report = validate_weather(_frame(rows))

    assert report["n_input"] == 5
    assert report["n_clean"] == 2
    assert report["n_rejected"] == 3
    assert report["rejected"] == {
        "null_key": 0,
        "null_measure": 0,
        "range": 1,         # the negative precip row
        "tmin_gt_tmax": 1,  # the inverted temps row
        "duplicate": 1,     # the repeated (Douala, 2023-01-01)
    }
    # The two surviving rows are the first two clean ones.
    assert clean.shape == (2, 6)
    assert list(clean["station"]) == ["Douala", "Douala"]
    assert list(clean["date"]) == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-02"),
    ]


def test_validate_null_key_and_measure() -> None:
    rows = [
        _good("Douala", "2023-01-01", 22.0, 31.0, 26.5, 0.0),
        _good(None, "2023-01-02", 23.0, 30.0, 26.0, 5.4),       # null station
        _good("Maroua", "2023-01-03", None, 30.0, 26.0, 1.0),   # null measure
    ]
    clean, report = validate_weather(_frame(rows))
    assert report["rejected"]["null_key"] == 1
    assert report["rejected"]["null_measure"] == 1
    assert report["n_clean"] == 1


def test_validate_out_of_range_temperature() -> None:
    rows = [
        _good("Garoua", "2023-01-01", -70.0, 10.0, -30.0, 0.0),  # tmin < -60
        _good("Garoua", "2023-01-02", 20.0, 80.0, 50.0, 0.0),    # tmax > 60
    ]
    clean, report = validate_weather(_frame(rows))
    assert report["rejected"]["range"] == 2
    assert report["n_clean"] == 0


def test_validate_empty_frame_is_fully_valid() -> None:
    empty = _frame(
        [{"station": "x", "date": "2023-01-01", "tmin_c": 1.0, "tmax_c": 2.0,
          "tmean_c": 1.5, "precip_mm": 0.0}]
    ).iloc[0:0]
    clean, report = validate_weather(empty)
    assert report["n_input"] == 0
    assert report["pct_valid"] == 1.0
    assert len(clean) == 0


def test_validate_pct_valid_fraction() -> None:
    rows = [
        _good("Douala", "2023-01-01", 22.0, 31.0, 26.5, 0.0),
        _good("Douala", "2023-01-02", 40.0, 10.0, 26.0, 5.4),  # bad
    ]
    _, report = validate_weather(_frame(rows))
    assert report["pct_valid"] == 0.5
