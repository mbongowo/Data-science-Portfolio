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
import pytest

from recsys.als import (
    als_factorize,
    als_factorize_biased,
    als_implicit,
    predict,
    predict_biased,
)
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


# --------------------------------------------------------------------------- #
# Biased matrix factorisation
# --------------------------------------------------------------------------- #
def _additive_matrix() -> np.ndarray:
    r"""Build a purely additive ratings matrix R[i,j] = mu + bu[i] + bv[j].

    With mu = 3, user offsets bu = [-1, 0, 1, 2] and item offsets
    bv = [0.5, -0.5, 1.0], every cell is the sum of a global mean and the two
    offsets — there is *no* interaction (latent) structure at all. This is the
    regime the biased model is designed for: the unbiased factorisation has to
    burn latent dimensions approximating an additive surface, whereas the biased
    model represents it exactly with mu + bu + bv.
    """
    mu = 3.0
    bu = np.array([-1.0, 0.0, 1.0, 2.0])
    bv = np.array([0.5, -0.5, 1.0])
    return mu + bu[:, None] + bv[None, :]


def test_biased_beats_unbiased_on_additive_data() -> None:
    """On bias-dominated data the biased model reaches a lower RMSE.

    Both models see the same fully observed additive matrix at the same rank.
    The biased model factors the user/item offsets out explicitly, so it fits
    the additive surface far better than the plain factorisation at rank 1.
    """
    R = _additive_matrix()
    mask = np.ones_like(R, dtype=bool)

    mu, bu, bv, U, V = als_factorize_biased(
        R, mask, rank=1, reg=0.0, iters=40, seed=0
    )
    biased_recon = predict_biased(mu, bu, bv, U, V)
    biased_rmse = rmse(R.ravel(), biased_recon.ravel())

    U0, V0 = als_factorize(R, mask, rank=1, reg=0.0, iters=40, seed=0)
    plain_rmse = rmse(R.ravel(), predict(U0, V0).ravel())

    assert biased_rmse < plain_rmse
    # The biased model recovers the additive surface essentially exactly.
    assert biased_rmse < 1e-3


def test_biased_recovers_global_mean() -> None:
    """mu is fixed to the mean of the observed ratings (hand-checkable)."""
    R = _additive_matrix()
    mask = np.ones_like(R, dtype=bool)
    mu, _bu, _bv, _U, _V = als_factorize_biased(R, mask, rank=1, iters=1, seed=0)
    assert mu == pytest.approx(float(R.mean()), abs=1e-12)


def test_biased_shapes() -> None:
    """Returned components have the documented shapes."""
    R = _additive_matrix()
    mask = np.ones_like(R, dtype=bool)
    mu, bu, bv, U, V = als_factorize_biased(R, mask, rank=2, reg=0.1, iters=3, seed=0)
    assert isinstance(mu, float)
    assert bu.shape == (R.shape[0],)
    assert bv.shape == (R.shape[1],)
    assert U.shape == (R.shape[0], 2)
    assert V.shape == (R.shape[1], 2)


def test_biased_reproducible_with_fixed_seed() -> None:
    """Same seed => identical factorisation."""
    R = _additive_matrix()
    mask = np.ones_like(R, dtype=bool)
    out1 = als_factorize_biased(R, mask, rank=2, reg=0.1, iters=10, seed=3)
    out2 = als_factorize_biased(R, mask, rank=2, reg=0.1, iters=10, seed=3)
    np.testing.assert_array_equal(out1[3], out2[3])
    np.testing.assert_array_equal(out1[4], out2[4])


def test_biased_rejects_bad_input() -> None:
    """Shape / rank / iters / empty-mask validation."""
    R = _additive_matrix()
    good = np.ones_like(R, dtype=bool)
    with pytest.raises(ValueError):
        als_factorize_biased(R, np.ones((2, 2), dtype=bool), rank=1)
    with pytest.raises(ValueError):
        als_factorize_biased(R, good, rank=0)
    with pytest.raises(ValueError):
        als_factorize_biased(R, good, rank=1, iters=0)
    with pytest.raises(ValueError):
        als_factorize_biased(R, np.zeros_like(R, dtype=bool), rank=1)  # empty mask


def test_biased_handles_all_equal_ratings() -> None:
    """An all-equal matrix is fit by mu alone; RMSE is ~0 and biases ~0."""
    R = np.full((4, 3), 4.0)
    mask = np.ones_like(R, dtype=bool)
    mu, bu, bv, U, V = als_factorize_biased(R, mask, rank=2, reg=0.1, iters=20, seed=0)
    recon = predict_biased(mu, bu, bv, U, V)
    assert rmse(R.ravel(), recon.ravel()) < 1e-3
    assert mu == pytest.approx(4.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Implicit-feedback ALS (Hu-Koren-Volinsky)
# --------------------------------------------------------------------------- #
def _two_block_preference() -> tuple[np.ndarray, np.ndarray]:
    """Binary preference with two user groups liking disjoint item blocks.

    Users 0-1 interact with items 0-1; users 2-3 interact with items 2-3. The
    confidence is ``1 + alpha * P`` with alpha = 39 (so observed cells have
    confidence 40, unobserved cells confidence 1), the standard HKV mapping.
    A correct factorisation must score each group's own block above the other.
    """
    P = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    C = 1.0 + 39.0 * P
    return C, P


def test_implicit_separates_preference_blocks() -> None:
    """Learned scores are higher on each group's preferred block than off it."""
    C, P = _two_block_preference()
    U, V = als_implicit(C, P, rank=2, reg=0.1, iters=30, seed=0)
    scores = predict(U, V)
    # For every user, the mean score over their preferred (P==1) items must
    # exceed the mean over their non-preferred (P==0) items.
    for i in range(P.shape[0]):
        liked = scores[i, P[i] == 1.0].mean()
        disliked = scores[i, P[i] == 0.0].mean()
        assert liked > disliked


def test_implicit_recommends_correct_block_top() -> None:
    """The top-ranked item for each user lies in that user's preferred block."""
    C, P = _two_block_preference()
    U, V = als_implicit(C, P, rank=2, reg=0.1, iters=30, seed=0)
    scores = predict(U, V)
    for i in range(P.shape[0]):
        top_item = int(np.argmax(scores[i]))
        assert P[i, top_item] == 1.0


def test_implicit_shapes_and_reproducible() -> None:
    """Factor shapes and seed reproducibility."""
    C, P = _two_block_preference()
    U1, V1 = als_implicit(C, P, rank=2, reg=0.1, iters=5, seed=11)
    U2, V2 = als_implicit(C, P, rank=2, reg=0.1, iters=5, seed=11)
    assert U1.shape == (P.shape[0], 2)
    assert V1.shape == (P.shape[1], 2)
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(V1, V2)


def test_implicit_rejects_bad_input() -> None:
    """Shape / rank / iters validation."""
    C, P = _two_block_preference()
    with pytest.raises(ValueError):
        als_implicit(C, P[:2], rank=2)  # shape mismatch
    with pytest.raises(ValueError):
        als_implicit(C, P, rank=0)
    with pytest.raises(ValueError):
        als_implicit(C, P, rank=2, iters=0)


def test_implicit_all_zero_preference() -> None:
    """An all-zero preference matrix (no interactions) yields finite zero-ish scores.

    With P == 0 everywhere the target is zero, so the ridge solve drives the
    factors toward zero; the predicted scores must stay finite (edge case: a
    user/item with no positive feedback at all).
    """
    P = np.zeros((3, 3))
    C = np.ones_like(P)  # confidence 1 everywhere, no preference
    U, V = als_implicit(C, P, rank=2, reg=0.1, iters=10, seed=0)
    scores = predict(U, V)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 0.0, atol=1e-6)
