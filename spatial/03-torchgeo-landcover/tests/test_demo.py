"""End-to-end test of the GPU-free demo. numpy only."""

from __future__ import annotations

import json

from lcnet.demo import run_demo


def test_run_demo_pins_key_metrics(tmp_path):
    result = run_demo(seed=0, out_dir=str(tmp_path))

    # Shape of the synthetic problem.
    assert result["n_samples"] == 480
    assert result["n_classes"] == 6
    assert result["n_features"] == 12  # 6 bands x [mean, std]

    # Strong but realistic accuracy (not a trivial 1.0).
    assert 0.70 < result["test_accuracy"] < 0.99
    assert 0.70 < result["test_macro_f1"] < 0.99

    # Pin the canonical seed-0 numbers.
    assert abs(result["test_accuracy"] - 0.8854166666666666) < 1e-6
    assert abs(result["test_macro_f1"] - 0.8815235690235691) < 1e-6

    # per_class_f1 has one entry per class, each in [0, 1].
    assert len(result["per_class_f1"]) == 6
    assert all(0.0 <= v <= 1.0 for v in result["per_class_f1"])


def test_run_demo_writes_artifacts(tmp_path):
    run_demo(seed=0, out_dir=str(tmp_path))

    metrics_path = tmp_path / "metrics.json"
    cm_path = tmp_path / "confusion_matrix.csv"
    assert metrics_path.exists()
    assert cm_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["n_classes"] == 6

    # Confusion matrix CSV: header + K rows, each with K count columns.
    lines = cm_path.read_text().strip().splitlines()
    assert len(lines) == 1 + 6  # header + 6 class rows
    for row in lines[1:]:
        cells = row.split(",")
        assert len(cells) == 1 + 6  # row label + 6 columns


def test_run_demo_deterministic(tmp_path):
    a = run_demo(seed=0, out_dir=str(tmp_path / "a"))
    b = run_demo(seed=0, out_dir=str(tmp_path / "b"))
    assert a == b
