r"""Known-answer tests for TF-IDF.

The corpus is ["good good", "bad"], so N = 2 and the sorted vocabulary is
['bad', 'good']. Each term appears in exactly one document, so df = 1 for both
and the smoothed idf is the same for both:

    idf = ln((N + 1) / (df + 1)) + 1 = ln(3 / 2) + 1

The TF-IDF weights (tf * idf, no row normalisation) are therefore:

    doc 0 "good good": good count 2 -> 2 * (ln 1.5 + 1); bad -> 0
    doc 1 "bad":       bad  count 1 -> 1 * (ln 1.5 + 1); good -> 0
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentiment.topics import nmf, tfidf

_IDF = math.log(3.0 / 2.0) + 1.0  # ln(1.5) + 1


def test_vocabulary_is_sorted() -> None:
    _, vocab = tfidf(["good good", "bad"])
    assert vocab == ["bad", "good"]


def test_tfidf_values_match_hand_derivation() -> None:
    matrix, vocab = tfidf(["good good", "bad"])
    assert matrix.shape == (2, 2)
    # columns are [bad, good]
    expected = np.array(
        [
            [0.0, 2.0 * _IDF],
            [1.0 * _IDF, 0.0],
        ]
    )
    assert matrix == pytest.approx(expected, abs=1e-12)


def test_term_in_every_document_has_idf_one() -> None:
    # "x" appears in both docs: idf = ln((2+1)/(2+1)) + 1 = 1, so weight == tf.
    matrix, vocab = tfidf(["x y", "x"])
    j = vocab.index("x")
    assert matrix[0, j] == pytest.approx(1.0, abs=1e-12)  # tf 1 * idf 1
    assert matrix[1, j] == pytest.approx(1.0, abs=1e-12)


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError):
        tfidf([])


# --- NMF topic modelling -------------------------------------------------


def _recon_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.linalg.norm(X - W @ H))


def test_nmf_factors_are_non_negative() -> None:
    X = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    W, H = nmf(X, k=2, iters=300, seed=0)
    assert W.shape == (2, 2)
    assert H.shape == (2, 4)
    assert (W >= 0).all()
    assert (H >= 0).all()


def test_nmf_reconstructs_a_block_matrix() -> None:
    # Two topics with disjoint vocabularies; each document is a non-negative
    # multiple of one topic, so X is exactly rank 2 and a 2-topic NMF
    # reconstructs it almost exactly. Comparable per-topic magnitudes keep the
    # multiplicative updates from collapsing a latent factor.
    topic_a = np.array([2.0, 1.0, 0.0, 0.0])
    topic_b = np.array([0.0, 0.0, 2.0, 1.0])
    X = np.array(
        [
            1.0 * topic_a,
            2.0 * topic_a,
            1.0 * topic_b,
            2.0 * topic_b,
        ]
    )
    W, H = nmf(X, k=2, iters=500, seed=0)
    assert _recon_error(X, W, H) < 0.1


def test_nmf_reconstruction_error_decreases() -> None:
    # More iterations must not worsen the fit (multiplicative updates are
    # monotone non-increasing on the Frobenius error).
    X = np.array(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 4.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 5.0, 0.0, 1.0],
        ]
    )
    err_few = _recon_error(X, *nmf(X, k=2, iters=10, seed=1))
    err_many = _recon_error(X, *nmf(X, k=2, iters=400, seed=1))
    assert err_many <= err_few


def test_nmf_k_one_edge_case() -> None:
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # rank-1, k=1 suffices
    W, H = nmf(X, k=1, iters=400, seed=0)
    assert W.shape == (3, 1)
    assert H.shape == (1, 2)
    assert (W >= 0).all() and (H >= 0).all()
    assert _recon_error(X, W, H) < 0.05


def test_nmf_is_reproducible_for_fixed_seed() -> None:
    X = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    W1, H1 = nmf(X, k=2, iters=100, seed=5)
    W2, H2 = nmf(X, k=2, iters=100, seed=5)
    assert np.array_equal(W1, W2)
    assert np.array_equal(H1, H2)


def test_nmf_input_guards() -> None:
    with pytest.raises(ValueError):
        nmf(np.array([[-1.0, 0.0]]), k=1)  # negative entry
    with pytest.raises(ValueError):
        nmf(np.array([[1.0, 0.0]]), k=0)  # k < 1
    with pytest.raises(ValueError):
        nmf(np.array([1.0, 2.0]), k=1)  # not 2-D
