"""Deterministic end-to-end test for the seeded synthetic demo.

``run_demo(seed=0)`` synthesizes a fixed labelled log, drives the real
templating + PCA detection + metrics core, and must reproduce exactly the
numbers quoted in the README. These asserts pin those committed values so the
"Result first" block can never silently drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from loganomaly.demo import run_demo


def test_run_demo_is_deterministic(tmp_path: Path) -> None:
    """Seed 0 reproduces the committed metrics and writes the artefacts."""
    result = run_demo(seed=0, out_dir=tmp_path)

    # Corpus shape.
    assert result["num_sessions"] == 300
    assert result["num_templates"] == 8
    assert result["num_anomalies_true"] == 45

    # Confusion matrix (positive class = anomaly). Sums to num_sessions.
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (244, 11, 14, 31)
    assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == 300

    # Metrics — believable, non-trivial separation (not a perfect 1.0).
    assert result["precision"] == 31 / (31 + 11)
    assert result["recall"] == 31 / (31 + 14)
    assert abs(result["f1"] - 0.7126436781609196) < 1e-12

    # Artefacts written.
    for name in ("templates.csv", "scores.csv", "summary.json"):
        assert (tmp_path / name).is_file()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["f1"] == result["f1"]


def test_run_demo_repeatable() -> None:
    """Calling twice with the same seed yields identical metrics."""
    import tempfile

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = run_demo(seed=0, out_dir=d1)
        b = run_demo(seed=0, out_dir=d2)
    assert a == b
