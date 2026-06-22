"""Tests for the dependency-free synthetic ESDA demo.

The demo plants a High-High block and a Low-Low block into a noisy 12 x 12 grid
and drives the pure-numpy core. With a fixed seed the statistics are
deterministic, so we can pin the headline numbers and confirm the planted
clustering shows up: a clearly positive global Moran's I, Geary's C well below
1, and non-zero HH / LL counts. These tests need only numpy + stdlib.
"""

from __future__ import annotations

import json

import pytest

from hotspots.demo import run_demo


def test_demo_metrics_are_pinned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Seed=0 yields the exact metrics quoted in the README."""
    out = run_demo(seed=0, out_dir=tmp_path)

    assert out["n"] == 144
    assert out["morans_i"] == pytest.approx(0.7241615642042645, abs=1e-9)
    assert out["expected_i"] == pytest.approx(-1.0 / 143.0, abs=1e-12)
    assert out["gearys_c"] == pytest.approx(0.23922733255116607, abs=1e-9)
    assert out["lisa_counts"] == {"HH": 48, "LL": 37, "LH": 33, "HL": 26, "ns": 0}
    assert out["gi_hot"] == 16
    assert out["gi_cold"] == 16


def test_demo_shows_planted_clustering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The planted structure must produce strong positive autocorrelation."""
    out = run_demo(seed=0, out_dir=tmp_path)

    # Moran's I clearly positive and well above its null expectation.
    assert out["morans_i"] > 0.5
    assert out["morans_i"] > out["expected_i"]
    # Geary's C below 1 is the companion signal of positive autocorrelation.
    assert out["gearys_c"] < 1.0
    # Both planted clusters register.
    assert out["lisa_counts"]["HH"] > 0
    assert out["lisa_counts"]["LL"] > 0
    # Gi* finds both a hot and a cold pocket.
    assert out["gi_hot"] > 0
    assert out["gi_cold"] > 0


def test_demo_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The demo writes summary.json and lisa_labels.csv, and they round-trip."""
    out = run_demo(seed=0, out_dir=tmp_path)

    summary_path = tmp_path / "summary.json"
    labels_path = tmp_path / "lisa_labels.csv"
    assert summary_path.exists()
    assert labels_path.exists()

    with open(summary_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["morans_i"] == pytest.approx(out["morans_i"], abs=1e-12)

    # One header row plus one row per grid cell.
    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 144 + 1


def test_demo_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Same seed -> identical metrics on a second run."""
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
