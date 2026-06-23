"""Determinism and correctness tests for the synthetic end-to-end demo.

The demo drives the real numpy core (features, logistic regression, metrics, PSI
/ KS drift) on seeded synthetic weather, so for a fixed seed every reported
number is reproducible bit-for-bit. These tests pin the committed values and
assert the two headline claims the README makes: the model is meaningfully
better than chance, and the planted distribution shift is caught.
"""

from __future__ import annotations

from pathlib import Path

from mlpipe.demo import run_demo

# Committed values produced by run_demo(seed=0) on Python 3.14 / numpy 2.4.4 /
# pandas 3.0.3. If the core or the synthesis changes, these update together.
EXPECTED = {
    "n_train": 714,
    "n_test": 179,
    "test_accuracy": 0.8939,
    "test_f1": 0.6667,
    "test_roc_auc": 0.9379,
    "n_features_drifted": 10,
}


def test_run_demo_metrics_are_deterministic(tmp_path: Path) -> None:
    """run_demo(seed=0) reproduces the committed metric values exactly."""
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["n_train"] == EXPECTED["n_train"]
    assert result["n_test"] == EXPECTED["n_test"]
    assert result["test_accuracy"] == EXPECTED["test_accuracy"]
    assert result["test_f1"] == EXPECTED["test_f1"]
    assert result["test_roc_auc"] == EXPECTED["test_roc_auc"]
    assert result["n_features_drifted"] == EXPECTED["n_features_drifted"]


def test_model_beats_chance(tmp_path: Path) -> None:
    """The rain-day classifier clears a non-trivial accuracy and ROC-AUC bar."""
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["test_accuracy"] > 0.6
    assert result["test_roc_auc"] > 0.7


def test_planted_drift_is_detected(tmp_path: Path) -> None:
    """The planted warm/wet shift flags at least one feature as drifted."""
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["n_features_drifted"] >= 1
    assert result["max_psi"] >= 0.2  # past the major-shift threshold
    assert len(result["drifted_features"]) == result["n_features_drifted"]


def test_run_demo_writes_artifacts(tmp_path: Path) -> None:
    """The two artifacts are written to the output directory."""
    run_demo(seed=0, out_dir=tmp_path)
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "drift_report.csv").is_file()
