"""Tests for the pure-numpy SoftmaxClassifier and standardize. numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from lcnet.classifier import SoftmaxClassifier, standardize


def _separable_three_class(seed: int = 0):
    """Three tight, well-separated Gaussian blobs in 2-D -> linearly separable."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    X_parts, y_parts = [], []
    for cls, c in enumerate(centers):
        X_parts.append(c + rng.normal(0.0, 0.3, size=(40, 2)))
        y_parts.append(np.full(40, cls))
    return np.vstack(X_parts), np.concatenate(y_parts)


def test_reaches_full_train_accuracy_on_separable():
    X, y = _separable_three_class()
    clf = SoftmaxClassifier().fit(X, y, lr=0.5, epochs=500, seed=0)
    acc = float(np.mean(clf.predict(X) == y))
    assert acc >= 0.99


def test_loss_decreases_monotonically():
    X, y = _separable_three_class()
    clf = SoftmaxClassifier().fit(X, y, lr=0.3, epochs=200, seed=0)
    losses = np.asarray(clf.loss_history)
    # Full-batch GD on a convex loss with a sane step -> non-increasing.
    diffs = np.diff(losses)
    assert np.all(diffs <= 1e-9)
    assert losses[-1] < losses[0]


def test_predict_proba_rows_sum_to_one():
    X, y = _separable_three_class()
    clf = SoftmaxClassifier().fit(X, y, epochs=50, seed=0)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0.0)


def test_reproducible_with_fixed_seed():
    X, y = _separable_three_class()
    a = SoftmaxClassifier().fit(X, y, epochs=100, seed=7)
    b = SoftmaxClassifier().fit(X, y, epochs=100, seed=7)
    assert np.array_equal(a.W, b.W)
    assert a.loss_history == b.loss_history


def test_standardize_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(200, 4))
    X_std, mean, std = standardize(X)
    assert np.allclose(X_std.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(X_std.std(axis=0), 1.0, atol=1e-9)
    assert mean.shape == (4,) and std.shape == (4,)


def test_standardize_uses_provided_stats():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])
    X_std, m, s = standardize(X, mean, std)
    # With identity stats the data is unchanged.
    assert np.array_equal(X_std, X)
    assert np.array_equal(m, mean)


def test_standardize_constant_column_no_nan():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    X_std, _, _ = standardize(X)
    assert not np.any(np.isnan(X_std))
    # Constant column maps to all zeros.
    assert np.allclose(X_std[:, 0], 0.0)


def test_fit_input_guards():
    clf = SoftmaxClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.zeros(5), np.zeros(5))  # X not 2-D
    with pytest.raises(ValueError):
        clf.fit(np.zeros((5, 2)), np.zeros(4))  # row mismatch
    with pytest.raises(ValueError):
        clf.fit(np.zeros((0, 2)), np.zeros(0))  # empty


def test_predict_before_fit_raises():
    clf = SoftmaxClassifier()
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((2, 2)))
