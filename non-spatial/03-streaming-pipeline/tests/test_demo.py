"""Deterministic test for the seeded synthetic air-quality demo.

The demo drives the *real* pure-Python core (tumbling windows -> EPA AQI ->
alert engine with cooldown) over a seeded synthetic stream with a planted
pollution spike, so its metrics are fully reproducible. Pinning the exact values
here keeps the README figures honest and guarantees the demo runs in CI. Only
numpy/pandas + stdlib are required (no Kafka/Spark).
"""

from __future__ import annotations

import json

from aqstream.demo import run_demo


def test_run_demo_is_deterministic(tmp_path) -> None:
    """run_demo(seed=0) returns the exact committed metrics (also in README)."""
    metrics = run_demo(seed=0, out_dir=tmp_path)

    assert metrics["seed"] == 0
    assert metrics["n_readings"] == 384
    assert metrics["n_stations"] == 4
    assert metrics["n_alerts"] == 51
    assert metrics["alerts_suppressed_by_cooldown"] == 187
    assert metrics["peak_aqi"] == 173
    assert metrics["peak_station"] == "Garoua"
    assert metrics["worst_category"] == "Unhealthy"


def test_run_demo_fires_and_suppresses() -> None:
    """The planted spike must fire alerts and the cooldown must suppress repeats."""
    metrics = run_demo(seed=0, out_dir="outputs")
    assert metrics["n_alerts"] >= 1
    assert metrics["alerts_suppressed_by_cooldown"] >= 1

    worst = metrics["worst_category"]
    rank = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    assert rank.index(worst) >= rank.index("Unhealthy for Sensitive Groups")


def test_run_demo_writes_artifacts(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)
    assert (tmp_path / "alerts.csv").is_file()
    assert (tmp_path / "hourly_aqi.csv").is_file()
    summary = tmp_path / "summary.json"
    assert summary.is_file()
    loaded = json.loads(summary.read_text(encoding="utf-8"))
    assert loaded["n_readings"] == 384


def test_run_demo_stable_across_calls(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
