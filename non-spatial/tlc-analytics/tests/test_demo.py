"""Tests for the reproducible pandas demo.

The demo drives the *real* core (``clean_trips`` then the marts) over a seeded
synthetic frame, so its headline numbers are deterministic and can be pinned.
These tests assert the committed insight values, that the cleaner drops exactly
the planted bad rows, and that the card tip rate beats cash (the core insight).
"""

from __future__ import annotations

import json

import pytest

from tlc.demo import N_BAD_ROWS, N_GOOD_TRIPS, run_demo


def test_run_demo_committed_metrics(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=str(tmp_path))

    # Row accounting: every planted bad row is dropped, all good rows survive.
    assert result["rows_in"] == N_GOOD_TRIPS + N_BAD_ROWS
    assert result["rows_after_clean"] == N_GOOD_TRIPS
    assert result["rows_in"] - result["rows_after_clean"] == N_BAD_ROWS

    # Pinned insight metrics (seed=0).
    assert result["peak_demand_hour"] == 18
    assert result["tip_rate_card"] == pytest.approx(0.1788582427366196, rel=1e-9)
    assert result["tip_rate_cash"] == pytest.approx(0.0, abs=1e-12)
    assert result["mean_fare"] == pytest.approx(26.493828000000004, rel=1e-9)

    # The real benchmark harness produced a non-negative timing.
    assert result["pandas_mart_build_seconds"] >= 0.0


def test_card_tip_rate_beats_cash(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=str(tmp_path))
    assert result["tip_rate_card"] > result["tip_rate_cash"]


def test_run_demo_writes_artifacts(tmp_path) -> None:
    run_demo(seed=0, out_dir=str(tmp_path))
    for name in (
        "hourly_demand.csv",
        "tip_rate_by_payment.csv",
        "fare_summary.csv",
        "summary.json",
    ):
        assert (tmp_path / name).is_file()

    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["peak_demand_hour"] == 18


def test_run_demo_is_deterministic(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=str(tmp_path / "a"))
    b = run_demo(seed=0, out_dir=str(tmp_path / "b"))
    a.pop("pandas_mart_build_seconds")
    b.pop("pandas_mart_build_seconds")
    assert a == b
