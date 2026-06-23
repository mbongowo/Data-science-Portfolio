"""Tests for the pure-numpy SoftmaxClassifier and the z-score scaler."""

from __future__ import annotations

import numpy as np
import pytest

from croprec.model import SoftmaxClassifier, standardize


def _separable_blobs(seed: int = 0):
    """Three well-separated 2-D Gaussian blobs -> near-perfect separability."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X_parts, y_parts = [], []
    for c, center in enumerate(centers):
        X_parts.append(center + 0.3 * rng.standard_normal((40, 2)))
        y_parts.append(np.full(40, c))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def test_fits_separable_to_near_perfect():
    X, y = _separable_blobs()
    Xs, _, _ = standardize(X)
    clf = SoftmaxClassifier().fit(Xs, y, lr=0.5, epochs=400, l2=1e-4, seed=0)
    assert (clf.predict(Xs) == y).mean() >= 0.99


def test_predict_proba_rows_sum_to_one():
    X, y = _separable_blobs()
    Xs, _, _ = standardize(X)
    clf = SoftmaxClassifier().fit(Xs, y, seed=0)
    proba = clf.predict_proba(Xs)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_top_k_returns_k_sorted_classes():
    X, y = _separable_blobs()
    Xs, _, _ = standardize(X)
    clf = SoftmaxClassifier().fit(Xs, y, seed=0)
    classes, probs = clf.top_k(Xs, k=2)
    assert classes.shape == (X.shape[0], 2)
    assert probs.shape == (X.shape[0], 2)
    # Each row's probabilities are in descending order.
    assert np.all(probs[:, 0] >= probs[:, 1])
    # The top-1 class equals argmax of predict.
    assert np.array_equal(classes[:, 0], clf.predict(Xs))


def test_top_k_clips_to_n_classes():
    X, y = _separable_blobs()
    Xs, _, _ = standardize(X)
    clf = SoftmaxClassifier().fit(Xs, y, seed=0)
    classes, probs = clf.top_k(Xs, k=10)  # only 3 classes exist
    assert classes.shape[1] == 3
    assert probs.shape[1] == 3


def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    X = rng.normal(5.0, 3.0, size=(200, 4))
    Xs, mean, std = standardize(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-9)
    assert mean.shape == (4,)
    assert std.shape == (4,)


def test_standardize_reuses_training_stats():
    X = np.array([[0.0], [2.0], [4.0]])
    Xs, mean, std = standardize(X)
    # Reapplying with the same stats to the same data reproduces the transform.
    Xs2, _, _ = standardize(X, mean=mean, std=std)
    assert np.allclose(Xs, Xs2)


def test_standardize_constant_column_safe():
    X = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    Xs, _, _ = standardize(X)
    assert np.all(np.isfinite(Xs))
    assert np.allclose(Xs[:, 0], 0.0)


def test_reproducible_seed():
    X, y = _separable_blobs()
    Xs, _, _ = standardize(X)
    a = SoftmaxClassifier().fit(Xs, y, seed=7)
    b = SoftmaxClassifier().fit(Xs, y, seed=7)
    assert np.allclose(a.W, b.W)
    assert a.loss_history == b.loss_history


def test_loss_decreases():
    X, y = _separable_blobs()
    Xs, _, _ = standardize(X)
    clf = SoftmaxClassifier().fit(Xs, y, seed=0, epochs=200)
    assert clf.loss_history[-1] < clf.loss_history[0]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SoftmaxClassifier().predict(np.zeros((2, 2)))


def test_fit_guards():
    clf = SoftmaxClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.zeros((0, 2)), np.array([]))
    with pytest.raises(ValueError):
        clf.fit(np.zeros((3, 2)), np.zeros(3))  # single class
    with pytest.raises(ValueError):
        clf.fit(np.zeros((3, 2)), np.array([0, 1]))  # mismatched rows
