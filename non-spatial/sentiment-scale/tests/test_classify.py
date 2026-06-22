r"""Tests for the pure-numpy logistic-regression classifier and bag-of-words.

Properties pinned here (no third-party ML in the tested path):

* **bag_of_words** is a known-answer count matrix over a sorted vocabulary, and
  honours a supplied vocabulary (so test docs encode against the train vocab).
* **LogisticRegression** drives its cross-entropy loss down monotonically and
  reaches **100% train accuracy on a linearly separable toy set** — the headline
  property of a working trained classifier.
* ``predict_proba`` returns valid probabilities; ``predict`` thresholds them.
* Edge cases: a single-class training set, an empty document, and the guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentiment.classify import LogisticRegression, bag_of_words


def test_bag_of_words_known_answer() -> None:
    X, vocab = bag_of_words(["good good bad", "bad"])
    assert vocab == ["bad", "good"]
    # row 0: bad=1, good=2 ; row 1: bad=1, good=0
    assert X.tolist() == [[1.0, 2.0], [1.0, 0.0]]


def test_bag_of_words_honours_supplied_vocab() -> None:
    # Encode a test doc against a fixed vocabulary; unknown tokens are dropped.
    X, vocab = bag_of_words(["good unseen great"], vocab=["bad", "good"])
    assert vocab == ["bad", "good"]
    assert X.tolist() == [[0.0, 1.0]]  # only "good" is in the vocab


def test_bag_of_words_empty_corpus_raises() -> None:
    with pytest.raises(ValueError):
        bag_of_words([])


def test_separable_set_reaches_perfect_train_accuracy() -> None:
    # Two clearly separable bag-of-words documents: a positive and a negative
    # class with disjoint vocabularies -> linearly separable -> accuracy 1.0.
    pos = ["great great good", "good great wonderful", "great good good"]
    neg = ["bad bad terrible", "terrible awful bad", "bad terrible awful"]
    docs = pos + neg
    y = np.array([1, 1, 1, 0, 0, 0])

    X, vocab = bag_of_words(docs)
    clf = LogisticRegression(lr=0.5, n_iters=2000).fit(X, y)

    acc = float((clf.predict(X) == y).mean())
    assert acc == 1.0


def test_loss_is_monotone_non_increasing() -> None:
    X = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 3.0]])
    y = np.array([1, 0, 1, 0])
    clf = LogisticRegression(lr=0.3, n_iters=500).fit(X, y)
    losses = np.array(clf.loss_history_)
    # Allow a tiny numerical tolerance; the loss must not climb.
    assert np.all(np.diff(losses) <= 1e-9)
    assert losses[-1] < losses[0]


def test_predict_proba_in_unit_interval_and_ordered() -> None:
    X = np.array([[2.0, 0.0], [0.0, 2.0]])
    y = np.array([1, 0])
    clf = LogisticRegression(lr=0.5, n_iters=1000).fit(X, y)
    proba = clf.predict_proba(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    # The positive example must get the higher probability.
    assert proba[0] > proba[1]


def test_predict_proba_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(np.zeros((1, 2)))


def test_single_class_training_set() -> None:
    # All-positive labels: the model should learn to predict class 1 everywhere
    # (degenerate but must not crash, and accuracy is trivially 1.0).
    X, _ = bag_of_words(["good great", "great good", "good good"])
    y = np.array([1, 1, 1])
    clf = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    assert clf.predict(X).tolist() == [1, 1, 1]


def test_empty_document_yields_zero_row() -> None:
    # An empty/punctuation-only doc tokenises to nothing -> all-zero feature row,
    # which the classifier scores at the bias-only probability without error.
    X, vocab = bag_of_words(["good", "   ", "..."])
    assert X[1].sum() == 0.0
    assert X[2].sum() == 0.0
    y = np.array([1, 0, 0])
    clf = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    proba = clf.predict_proba(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_fit_validates_shapes() -> None:
    clf = LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit(np.zeros(3), np.zeros(3))  # 1-D X
    with pytest.raises(ValueError):
        clf.fit(np.zeros((2, 2)), np.zeros(3))  # mismatched lengths
    with pytest.raises(ValueError):
        clf.fit(np.zeros((0, 2)), np.zeros(0))  # empty
