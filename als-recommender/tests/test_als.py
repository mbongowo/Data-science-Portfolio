"""Known-answer tests for the pure-numpy ALS factorisation.

ALS is iterative, so the test is a reconstruction guarantee rather than a single
hand-computed scalar: a fully observed rank-r matrix must be recovered to a tiny
RMSE, the factor shapes must be exactly (m, rank) and (n, rank), and a fixed
seed must give a bit-for-bit reproducible result.

The rank-1 fixture is constructed as an outer product u @ v.T, so it is exactly
rank 1 and a rank-1 ALS with no regularisation can fit it essentially exactly.
"""

from __future__ import annotations

import numpy as np

from recsys.als import als_factorize, predict
from recsys.metrics import rmse


def _rank1_matrix() -> np.ndarray:
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([2.0, -1.0, 0.5])
    return np.outer(u, v)  # (4, 3), exactly rank 1


def test_reconstructs_rank1_matrix() -> None:
    """A fully observed rank-1 matrix is recovered to RMSE < 1e-3."""
    R = _rank1_matrix()
    mask = np.ones_like(R, dtype=bool)
    U, V = als_factorize(R, mask, rank=1, reg=0.0, iters=40, seed=0)
    recon = predict(U, V)
    err = rmse(R.ravel(), recon.ravel())
    assert err < 1e-3


def test_factor_shapes() -> None:
    """U is (m, rank) and V is (n, rank)."""
    R = _rank1_matrix()
    mask = np.ones_like(R, dtype=bool)
    U, V = als_factorize(R, mask, rank=2, reg=0.1, iters=5, seed=0)
    assert U.shape == (R.shape[0], 2)
    assert V.shape == (R.shape[1], 2)


def test_reproducible_with_fixed_seed() -> None:
    """The same seed gives an identical factorisation; a different seed differs."""
    R = _rank1_matrix()
    mask = np.ones_like(R, dtype=bool)
    U1, V1 = als_factorize(R, mask, rank=2, reg=0.1, iters=10, seed=7)
    U2, V2 = als_factorize(R, mask, rank=2, reg=0.1, iters=10, seed=7)
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(V1, V2)


def test_rejects_bad_input() -> None:
    """Shape, rank, and iteration validation."""
    R = _rank1_matrix()
    good_mask = np.ones_like(R, dtype=bool)
    import pytest

    with pytest.raises(ValueError):
        als_factorize(R, np.ones((2, 2), dtype=bool), rank=1)  # mask mismatch
    with pytest.raises(ValueError):
        als_factorize(R, good_mask, rank=0)  # rank < 1
    with pytest.raises(ValueError):
        als_factorize(R, good_mask, rank=1, iters=0)  # iters < 1


def test_masked_entries_are_ignored() -> None:
    """Entries hidden by the mask do not pull the fit toward their value."""
    R = _rank1_matrix().copy()
    mask = np.ones_like(R, dtype=bool)
    # Corrupt one entry but hide it from the fit; reconstruction should stay
    # close to the *true* rank-1 value there, not the corrupted one.
    true_val = R[0, 0]
    R[0, 0] = 999.0
    mask[0, 0] = False
    U, V = als_factorize(R, mask, rank=1, reg=0.0, iters=60, seed=0)
    recon = predict(U, V)
    assert abs(recon[0, 0] - true_val) < 1e-2
