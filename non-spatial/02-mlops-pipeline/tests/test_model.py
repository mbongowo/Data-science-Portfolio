"""Tests for the pure-numpy logistic-regression classifier and the scaler.

These assert the learning behaviour a reviewer expects: the model separates a
clearly separable toy set, its loss decreases, its probabilities are valid, the
scaler does what it claims, and a fixed seed is reproducible. numpy only, so they
always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlpipe.model import LogisticRegression, standardize


def _separable_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs in 2-D, labelled 0 / 1."""
    rng = np.random.default_rng(seed)
    half = n // 2
    neg = rng.normal(-2.0, 0.5, size=(half, 2))
    pos = rng.normal(2.0, 0.5, size=(half, 2))
    X = np.vstack([neg, pos])
    y = np.concatenate([np.zeros(half), np.ones(half)])
    return X, y


def test_fits_separable_data_near_perfectly() -> None:
    """On clearly separable blobs, training accuracy is ~1.0."""
    X, y = _separable_data()
    model = LogisticRegression(lr=0.1, epochs=500, seed=0).fit(X, y)
    preds = model.predict(X)
    assert float(np.mean(preds == y)) >= 0.99


def test_loss_decreases() -> None:
    """The recorded cross-entropy is non-increasing and ends below it started."""
    X, y = _separable_data()
    model = LogisticRegression(lr=0.1, epochs=300, seed=0).fit(X, y)
    history = np.array(model.loss_history_)
    assert history[-1] < history[0]
    # Allow tiny numerical wiggle but require an overall downward trend.
    assert np.all(np.diff(history) <= 1e-6)


def test_predict_proba_in_unit_interval() -> None:
    """Probabilities stay within [0, 1]."""
    X, y = _separable_data()
    model = LogisticRegression(lr=0.1, epochs=200, seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_reproducible_seed() -> None:
    """Same seed => identical learned weights; different seed may differ."""
    X, y = _separable_data()
    a = LogisticRegression(lr=0.1, epochs=100, seed=7).fit(X, y)
    b = LogisticRegression(lr=0.1, epochs=100, seed=7).fit(X, y)
    assert np.allclose(a.weights_, b.weights_)
    assert a.bias_ == pytest.approx(b.bias_)


def test_standardize_zero_mean_unit_std() -> None:
    """Standardized columns have ~0 mean and ~1 std."""
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(500, 4))
    Xs, mu, sd = standardize(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-9)
    assert mu.shape == (4,)
    assert sd.shape == (4,)


def test_standardize_reuses_supplied_stats() -> None:
    """Passing train statistics applies them verbatim (no re-fit on new data)."""
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 1.0, size=(100, 3))
    _, mu, sd = standardize(train)
    new = rng.normal(0.0, 1.0, size=(10, 3))
    scaled, mu2, sd2 = standardize(new, mean=mu, std=sd)
    assert np.allclose(mu, mu2)
    assert np.allclose(sd, sd2)
    assert np.allclose(scaled, (new - mu) / sd)


def test_standardize_handles_zero_variance_column() -> None:
    """A constant column is divided by 1, not 0 (no NaN/inf)."""
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    Xs, _, sd = standardize(X)
    assert np.all(np.isfinite(Xs))
    assert sd[0] == 1.0  # zero std replaced by 1


def test_predict_proba_before_fit_raises() -> None:
    with pytest.raises(ValueError):
        LogisticRegression().predict_proba(np.zeros((2, 2)))


def test_fit_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros((3, 2)), np.zeros(2))


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros((0, 2)), np.zeros(0))
