"""Known-answer and edge-case tests for the Mahalanobis detector.

Hand-derived anchor (1-D, so the pseudo-inverse is just ``1 / variance``):

    x = [0, 0, 0, 0, 10]
    mean   = 2
    var    = mean((x-2)^2) = (4 + 4 + 4 + 4 + 64) / 5 = 16
    d_i^2  = (x_i - 2)^2 / 16

    => [0.25, 0.25, 0.25, 0.25, 4.0]

so the lone large value scores 4.0 and is the unique top score. These checks
also pin the robustness the pseudo-inverse buys: a zero-variance feature and two
perfectly collinear features must not blow the score up.
"""

from __future__ import annotations

import numpy as np

from loganomaly.detect import (
    flag,
    mahalanobis_scores,
    mahalanobis_threshold,
)


def test_mahalanobis_1d_known_answer() -> None:
    """Hand-worked 1-D case: scores are (x-mean)^2 / var."""
    x = np.array([[0.0], [0.0], [0.0], [0.0], [10.0]])
    scores = mahalanobis_scores(x)
    assert np.allclose(scores, [0.25, 0.25, 0.25, 0.25, 4.0])
    assert int(np.argmax(scores)) == 4


def test_mahalanobis_flags_the_outlier() -> None:
    """The single far-from-centre row clears a high quantile; nothing else does."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    X = np.vstack([X, np.array([20.0, -20.0, 20.0])])  # blatant outlier, index 40
    scores = mahalanobis_scores(X)
    assert int(np.argmax(scores)) == 40
    mask = mahalanobis_threshold(scores, quantile=0.95)
    assert mask[40]


def test_mahalanobis_threshold_matches_flag() -> None:
    """mahalanobis_threshold is exactly flag on the scores."""
    scores = np.array([0.1, 0.2, 5.0, 0.3])
    assert np.array_equal(mahalanobis_threshold(scores, 0.9), flag(scores, 0.9))


def test_mahalanobis_zero_variance_feature_is_stable() -> None:
    """A constant column contributes nothing and must not blow scores up.

    Columns 0,1 carry signal; column 2 is constant (zero variance). The
    pseudo-inverse ignores column 2, so the scores equal those of the 2-column
    matrix without it.
    """
    core = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [9.0, 9.0]])
    const_col = np.full((core.shape[0], 1), 7.0)
    X = np.hstack([core, const_col])

    scores_full = mahalanobis_scores(X)
    scores_core = mahalanobis_scores(core)
    assert np.all(np.isfinite(scores_full))
    assert np.allclose(scores_full, scores_core)


def test_mahalanobis_collinear_features_are_stable() -> None:
    """Two perfectly collinear columns make the covariance singular; pinv copes."""
    a = np.array([0.0, 1.0, 2.0, 3.0, 12.0])
    X = np.column_stack([a, 2.0 * a])  # column 1 is exactly 2x column 0
    scores = mahalanobis_scores(X)
    assert np.all(np.isfinite(scores))
    assert int(np.argmax(scores)) == 4


def test_mahalanobis_single_row_scores_zero() -> None:
    """One row => zero covariance => every score is 0 (nothing to be far from)."""
    scores = mahalanobis_scores(np.array([[3.0, 4.0, 5.0]]))
    assert scores.shape == (1,)
    assert np.allclose(scores, 0.0)


def test_mahalanobis_constant_matrix_scores_zero() -> None:
    """All rows identical => no spread => all scores 0."""
    X = np.full((6, 4), 2.5)
    assert np.allclose(mahalanobis_scores(X), 0.0)
