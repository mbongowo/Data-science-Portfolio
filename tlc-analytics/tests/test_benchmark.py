"""Known-answer tests for the benchmark harness (pure stdlib + pandas)."""

from __future__ import annotations

import pytest

from tlc.benchmark import BenchmarkResult, summarize, time_callable


def test_time_callable_returns_result_and_nonneg_duration() -> None:
    result, seconds = time_callable(lambda: 21 * 2)
    assert result == 42
    assert seconds >= 0.0


def test_time_callable_runs_fn_exactly_once() -> None:
    calls = []
    time_callable(lambda: calls.append(1))
    assert calls == [1]


def test_summarize_ranks_by_seconds() -> None:
    results = [
        BenchmarkResult(engine="spark", query="hourly_demand", seconds=3.0),
        BenchmarkResult(engine="duckdb", query="hourly_demand", seconds=1.0),
        BenchmarkResult(engine="warehouse", query="hourly_demand", seconds=2.0),
    ]
    out = summarize(results)
    # Fastest first.
    assert out["engine"].tolist() == ["duckdb", "warehouse", "spark"]
    assert out["rank"].tolist() == [1, 2, 3]
    assert out["seconds"].tolist() == [1.0, 2.0, 3.0]


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize([])
