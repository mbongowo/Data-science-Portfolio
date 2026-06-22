"""Tests for the real pandas-vs-DuckDB bake-off path.

The DuckDB path is run only when ``duckdb`` is importable (``pytest.importorskip``
guards it); the unavailable-skip behaviour is tested by monkeypatching the
availability probe so it runs even on a machine that *does* have duckdb.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tlc import benchmark
from tlc.benchmark import bake_off, duckdb_available, run_duckdb_query


def _clean_frame() -> pd.DataFrame:
    # Hand-derived hourly demand: hour 9 -> 3, hour 18 -> 2, hour 23 -> 1.
    return pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                [
                    "2023-01-02 09:00:00",
                    "2023-01-02 09:30:00",
                    "2023-01-02 09:45:00",
                    "2023-01-03 18:00:00",
                    "2023-01-03 18:45:00",
                    "2023-01-07 23:00:00",
                ]
            ),
        }
    )


def test_run_duckdb_query_matches_pandas_known_answer() -> None:
    pytest.importorskip("duckdb")
    out = run_duckdb_query(_clean_frame())
    got = {int(h): int(t) for h, t in zip(out["hour"], out["trips"], strict=True)}
    assert got == {9: 3, 18: 2, 23: 1}


def test_bake_off_measured_when_duckdb_present() -> None:
    pytest.importorskip("duckdb")
    ranking = bake_off(_clean_frame())
    engines = set(ranking["engine"])
    assert engines == {"pandas", "duckdb"}
    # Both engines were really measured.
    assert (ranking["status"] == "measured").all()
    # Non-negative timings for both.
    assert (ranking["seconds"] >= 0.0).all()


def test_bake_off_skips_cleanly_without_duckdb(monkeypatch) -> None:
    # Force the unavailable branch regardless of the local environment.
    monkeypatch.setattr(benchmark, "duckdb_available", lambda: False)
    ranking = bake_off(_clean_frame())
    pandas_row = ranking[ranking["engine"] == "pandas"].iloc[0]
    duckdb_row = ranking[ranking["engine"] == "duckdb"].iloc[0]
    assert pandas_row["status"] == "measured"
    assert duckdb_row["status"] == "engine unavailable"
    # The skipped engine has no timing.
    assert pd.isna(duckdb_row["seconds"])


def test_duckdb_available_is_boolean() -> None:
    assert isinstance(duckdb_available(), bool)
