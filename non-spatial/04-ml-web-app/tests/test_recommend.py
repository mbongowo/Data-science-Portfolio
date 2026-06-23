"""Tests for the single-sample recommendation flow."""

from __future__ import annotations

import numpy as np

from croprec.data import FEATURE_COLUMNS
from croprec.model import SoftmaxClassifier, standardize
from croprec.recommend import recommend


def _tiny_model(seed: int = 0):
    """Train a 3-crop model on synthetic rows with separated feature centres.

    Returns the fitted model plus the statistics ``recommend`` needs, and the
    centres so a test can sit a sample squarely in one crop's region.
    """
    centers = {
        "maize": np.array([80, 45, 40, 24, 65, 6.2, 90], dtype=float),
        "rice": np.array([90, 50, 40, 24, 82, 6.0, 220], dtype=float),
        "sorghum": np.array([60, 40, 35, 29, 45, 6.8, 55], dtype=float),
    }
    rng = np.random.default_rng(seed)
    X_parts, labels = [], []
    for crop, center in centers.items():
        X_parts.append(center + rng.normal(0, 3.0, size=(40, 7)))
        labels.extend([crop] * 40)
    X = np.vstack(X_parts)
    classes = np.array(sorted(centers))
    label_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_int[c] for c in labels])

    Xs, mean, std = standardize(X)
    model = SoftmaxClassifier().fit(Xs, y, seed=seed)
    return model, classes, mean, std, centers


def test_recommend_sorted_and_sums_to_one():
    model, classes, mean, std, centers = _tiny_model()
    sample = dict(zip(FEATURE_COLUMNS, centers["rice"], strict=True))
    ranked = recommend(model, classes, mean, std, sample)

    assert len(ranked) == 3
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 1e-9


def test_recommend_picks_the_right_crop():
    model, classes, mean, std, centers = _tiny_model()
    for crop, center in centers.items():
        sample = dict(zip(FEATURE_COLUMNS, center, strict=True))
        ranked = recommend(model, classes, mean, std, sample)
        assert ranked[0][0] == crop


def test_recommend_missing_feature_raises():
    model, classes, mean, std, _ = _tiny_model()
    bad = {"N": 80}  # missing the rest
    try:
        recommend(model, classes, mean, std, bad)
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected ValueError for missing features")
