"""Tests for the pure-python metric aggregation in ``geoseg.evaluate``.

The torch-dependent ``evaluate`` and matplotlib ``save_prediction_panel`` paths
are not exercised here; only ``aggregate_metrics`` and ``save_metrics`` are
pure-python.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geoseg.evaluate import aggregate_metrics, save_metrics


def test_aggregate_empty_returns_unit_scores():
    out = aggregate_metrics([], [])
    assert out == {"iou": 1.0, "f1": 1.0, "n": 0}


def test_aggregate_perfect_predictions():
    a = np.ones((2, 2), dtype=bool)
    out = aggregate_metrics([a, a], [a, a])
    assert out["n"] == 2
    assert abs(out["iou"] - 1.0) < 1e-6
    assert abs(out["f1"] - 1.0) < 1e-6


def test_aggregate_averages_over_pairs():
    perfect = np.array([[1, 1], [0, 0]], dtype=bool)
    half = np.array([[1, 0], [1, 0]], dtype=bool)
    half_target = np.array([[0, 0], [1, 1]], dtype=bool)  # IoU 1/3 vs half
    out = aggregate_metrics([perfect, half], [perfect, half_target])
    expected_iou = (1.0 + (1.0 / 3.0)) / 2.0
    assert abs(out["iou"] - expected_iou) < 1e-6


def test_aggregate_length_mismatch_raises():
    a = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        aggregate_metrics([a], [a, a])


def test_save_metrics_roundtrip(tmp_path):
    metrics = {"iou": 0.5, "f1": 0.6, "n": 3}
    path = save_metrics(metrics, tmp_path / "sub" / "metrics.json")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == metrics
