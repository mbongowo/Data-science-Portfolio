"""Tests for the dependency-free synthetic time-series + change demo.

The demo synthesises a multi-year NDVI stack with a planted clearing and drives
the pure-numpy core. With a fixed seed the metrics are deterministic, so we pin
the headline numbers quoted in the README and confirm the detector recovers the
planted clearing. These tests need only numpy + stdlib.
"""

from __future__ import annotations

import json

import pytest

from eets.demo import run_demo


def test_demo_metrics_are_pinned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Seed=0 yields the exact metrics quoted in the README."""
    out = run_demo(seed=0, out_dir=tmp_path)

    assert out["n_timesteps"] == 12
    assert out["loss_hectares"] == pytest.approx(9.0, abs=1e-9)
    assert out["gain_hectares"] == pytest.approx(0.0, abs=1e-9)
    assert out["planted_block_hectares"] == pytest.approx(9.0, abs=1e-9)
    assert out["baseline_mean_ndvi"] == pytest.approx(0.850063233503023, abs=1e-9)
    assert out["recent_mean_ndvi"] == pytest.approx(0.7655059080335063, abs=1e-9)
    assert out["pixel_size_m"] == pytest.approx(10.0)


def test_demo_recovers_planted_clearing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """NDVI drops in the recent period and the clearing is recovered as loss."""
    out = run_demo(seed=0, out_dir=tmp_path)
    assert out["recent_mean_ndvi"] < out["baseline_mean_ndvi"]
    assert out["planted_loss_recovered"] >= 0.8


def test_demo_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The demo writes change_stats.json and index_timeseries.csv, round-tripping."""
    out = run_demo(seed=0, out_dir=tmp_path)

    stats_path = tmp_path / "change_stats.json"
    series_path = tmp_path / "index_timeseries.csv"
    assert stats_path.exists()
    assert series_path.exists()

    with open(stats_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["loss_hectares"] == out["loss_hectares"]

    # Header row plus one row per scene (12 timesteps).
    lines = series_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12 + 1


def test_demo_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Same seed -> identical metrics on a second run."""
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
