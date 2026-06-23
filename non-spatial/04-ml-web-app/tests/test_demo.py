"""Deterministic test for the seeded synthetic crop-recommender demo.

The demo drives the real pure-numpy core (synthesise -> standardise -> softmax
classifier -> metrics) over a seeded synthetic dataset, so its metrics are fully
reproducible. Pinning the key values keeps the README figures honest and
guarantees the demo runs in CI. Only numpy/pandas + stdlib are required.
"""

from __future__ import annotations

import json

from croprec.demo import run_demo


def test_run_demo_pinned_metrics(tmp_path):
    metrics = run_demo(seed=0, out_dir=tmp_path)

    assert metrics["seed"] == 0
    assert metrics["n_samples"] == 800
    assert metrics["n_crops"] == 10
    assert metrics["n_features"] == 7
    assert metrics["test_accuracy"] == 0.8958
    assert metrics["test_macro_f1"] == 0.8942
    assert metrics["test_top3_accuracy"] == 0.9875


def test_run_demo_quality_bounds(tmp_path):
    metrics = run_demo(seed=0, out_dir=tmp_path)
    assert metrics["test_accuracy"] > 0.7
    assert metrics["test_top3_accuracy"] >= metrics["test_accuracy"]


def test_run_demo_writes_artifacts(tmp_path):
    run_demo(seed=0, out_dir=tmp_path)
    assert (tmp_path / "confusion_matrix.csv").is_file()
    summary = tmp_path / "metrics.json"
    assert summary.is_file()
    loaded = json.loads(summary.read_text(encoding="utf-8"))
    assert loaded["n_crops"] == 10


def test_run_demo_stable_across_calls(tmp_path):
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
