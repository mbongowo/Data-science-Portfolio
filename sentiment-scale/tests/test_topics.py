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

from sentiment.topics import tfidf

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
