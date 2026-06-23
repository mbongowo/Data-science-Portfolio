"""Known-answer tests for the pure-Python windowing helpers.

Hand-derived expected values on tiny inputs, no third-party engine.

Worked examples:

1. tumbling_aggregate. Readings at ts = [0, 1800, 3600, 5400] for station "A"
   with window_s = 3600 and value pm25:
     0, 1800 -> window 0;  3600, 5400 -> window 3600.
   mean of [10, 20] = 15 in window 0; mean of [30, 50] = 40 in window 3600.
   => {(0, "A"): 15.0, (3600, "A"): 40.0}.

2. rolling_mean. values = [1, 2, 3, 4], n = 2 ->
   [None, (1+2)/2, (2+3)/2, (3+4)/2] = [None, 1.5, 2.5, 3.5].

3. dedupe. keep first per key "id".
"""

from __future__ import annotations

import pytest

from aqstream import dedupe, rolling_mean, tumbling_aggregate


def _r(ts, station, pm25):
    return {"ts": ts, "station": station, "pm25": pm25}


def test_tumbling_aggregate_mean_known() -> None:
    readings = [
        _r(0, "A", 10.0),
        _r(1800, "A", 20.0),
        _r(3600, "A", 30.0),
        _r(5400, "A", 50.0),
    ]
    out = tumbling_aggregate(readings, 3600.0, "pm25", agg="mean")
    assert out == {(0.0, "A"): 15.0, (3600.0, "A"): 40.0}


def test_tumbling_aggregate_boundary_goes_to_later_window() -> None:
    """ts exactly on a boundary lands in the later, half-open window."""
    out = tumbling_aggregate([_r(3600, "A", 7.0)], 3600.0, "pm25")
    assert out == {(3600.0, "A"): 7.0}


def test_tumbling_aggregate_per_station() -> None:
    readings = [_r(0, "A", 10.0), _r(0, "B", 20.0), _r(100, "A", 30.0)]
    out = tumbling_aggregate(readings, 3600.0, "pm25", agg="max")
    assert out == {(0.0, "A"): 30.0, (0.0, "B"): 20.0}


def test_tumbling_aggregate_count() -> None:
    readings = [_r(0, "A", 10.0), _r(100, "A", 20.0), _r(200, "A", 30.0)]
    out = tumbling_aggregate(readings, 3600.0, "pm25", agg="count")
    assert out == {(0.0, "A"): 3.0}


def test_tumbling_aggregate_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        tumbling_aggregate([_r(0, "A", 1.0)], 0.0, "pm25")
    with pytest.raises(ValueError):
        tumbling_aggregate([_r(0, "A", 1.0)], 3600.0, "pm25", agg="median")
    with pytest.raises(ValueError):
        tumbling_aggregate([{"ts": 0, "pm25": 1.0}], 3600.0, "pm25")


def test_rolling_mean_known() -> None:
    assert rolling_mean([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


def test_rolling_mean_width_one_is_identity() -> None:
    assert rolling_mean([5, 6, 7], 1) == [5.0, 6.0, 7.0]


def test_rolling_mean_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        rolling_mean([1, 2, 3], 0)


def test_dedupe_keeps_first() -> None:
    readings = [
        {"id": 1, "v": "a"},
        {"id": 2, "v": "b"},
        {"id": 1, "v": "c"},
        {"id": 3, "v": "d"},
    ]
    out = dedupe(readings, "id")
    assert [r["v"] for r in out] == ["a", "b", "d"]


def test_dedupe_empty() -> None:
    assert dedupe([], "id") == []


def test_dedupe_rejects_missing_key() -> None:
    with pytest.raises(ValueError):
        dedupe([{"v": "a"}], "id")
