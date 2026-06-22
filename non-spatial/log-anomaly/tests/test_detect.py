"""Known-answer tests for the pure-numpy detectors.

These check structural properties that hold by construction, so a green test
proves the estimator behaves, not merely that it runs. No third-party dependency
beyond numpy.

1. PCA reconstruction error is ~0 when ``k`` reaches the rank of the centred
   matrix: the projection is then exact.
2. On a rank-1 matrix (every row a multiple of one direction) plus one row that
   breaks the pattern, the broken row has the largest reconstruction error at
   ``k = 1`` and is the one ``flag`` picks out.
3. ``zscore_anomalies`` flags exactly the entries above ``mean + z * std``.
"""

from __future__ import annotations

import numpy as np

from loganomaly.detect import flag, pca_reconstruction_error, zscore_anomalies


def test_pca_error_zero_when_k_is_rank() -> None:
    """When k >= rank of the centred matrix, reconstruction is exact (~0)."""
    rng = np.random.default_rng(0)
    # rank-2 data embedded in 5 columns: rows are combinations of two directions
    coeffs = rng.normal(size=(20, 2))
    basis = rng.normal(size=(2, 5))
    X = coeffs @ basis
    errors = pca_reconstruction_error(X, k=2)
    assert np.allclose(errors, 0.0, atol=1e-9)


def test_pca_error_decreases_with_k() -> None:
    """More components never increase the reconstruction error."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    e1 = pca_reconstruction_error(X, k=1).sum()
    e3 = pca_reconstruction_error(X, k=3).sum()
    assert e3 <= e1 + 1e-9


def test_pca_flags_the_off_pattern_row() -> None:
    """A rank-1 block plus one off-pattern row: the outlier has the top error."""
    # Nine rows along a single direction (rank 1), then one row off that line.
    base = np.array([1.0, 2.0, 3.0])
    X = np.vstack([base * c for c in (1, 2, 3, 4, 5, 6, 7, 8, 9)])
    X = np.vstack([X, np.array([9.0, 1.0, 1.0])])  # the off-pattern row (index 9)

    errors = pca_reconstruction_error(X, k=1)
    assert int(np.argmax(errors)) == 9

    mask = flag(errors, quantile=0.9)
    # Only the single off-pattern row clears the 0.9 quantile.
    assert mask[9]
    assert mask.sum() == 1


def test_flag_thresholds_above_quantile() -> None:
    """flag marks errors strictly above the requested quantile."""
    errors = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    mask = flag(errors, quantile=0.8)
    assert mask.tolist() == [False, False, False, False, True]


def test_zscore_anomalies_known() -> None:
    """One large value sits well above mean + z*std and is the only flag."""
    scores = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    mask = zscore_anomalies(scores, z=1.5)
    assert mask.tolist() == [False, False, False, False, True]


def test_zscore_zero_variance_flags_nothing() -> None:
    """A constant input has no spread, so nothing is anomalous."""
    mask = zscore_anomalies(np.array([5.0, 5.0, 5.0]), z=1.0)
    assert not mask.any()
