"""The reproducible demo must recover the planted disturbance (seed=0).

Pins the seed=0 metrics so the README numbers stay honest, and checks the
artefacts land on disk.
"""

from __future__ import annotations

import json

import pytest

from disturb.demo import run_demo


def test_demo_recovers_planted_break(tmp_path):
    out = run_demo(seed=0, out_dir=tmp_path)

    # Headline contract.
    assert out["n_obs"] == 115
    assert out["detected"] is True
    # The planted break is recovered at (essentially) the right location.
    assert abs(out["detected_index"] - out["planted_break_index"]) <= 2
    # A real drop: magnitude is negative.
    assert out["detected_magnitude"] < 0.0


def test_demo_metrics_are_pinned(tmp_path):
    out = run_demo(seed=0, out_dir=tmp_path)
    assert out["planted_break_index"] == 71
    assert out["detected_index"] == 71
    assert out["detected_magnitude"] == pytest.approx(-0.0991, abs=1e-3)
    assert out["detected_score"] == pytest.approx(2.699, abs=1e-2)
    assert out["seasonal_amplitude"] == pytest.approx(0.2264, abs=1e-3)


def test_demo_is_deterministic(tmp_path):
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b


def test_demo_writes_artifacts(tmp_path):
    run_demo(seed=0, out_dir=tmp_path)
    for name in ("series.csv", "components.csv", "summary.json"):
        assert (tmp_path / name).exists(), name

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_obs"] == 115
    assert summary["detected"] is True
    # components.csv has a header + one row per observation.
    lines = (tmp_path / "components.csv").read_text().strip().splitlines()
    assert len(lines) == summary["n_obs"] + 1
