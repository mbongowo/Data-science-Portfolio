"""Known-answer tests for the pure-numpy classification metrics."""

from __future__ import annotations

import numpy as np

from croprec.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_f1,
    top_k_accuracy,
)


def test_accuracy_known():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    assert accuracy(y_true, y_pred) == 0.75


def test_confusion_matrix_known():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, k=3)
    expected = np.array(
        [
            [1, 1, 0],  # true 0: one correct, one called 1
            [0, 2, 0],  # true 1: both correct
            [0, 0, 1],  # true 2: correct
        ]
    )
    assert np.array_equal(cm, expected)


def test_per_class_f1_known():
    # Class 0: TP=1, FP=0, FN=1 -> P=1, R=0.5, F1=2/3.
    # Class 1: TP=2, FP=1, FN=0 -> P=2/3, R=1, F1=0.8.
    # Class 2: TP=1, FP=0, FN=0 -> F1=1.
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    f1 = per_class_f1(y_true, y_pred, k=3)
    assert np.allclose(f1, [2.0 / 3.0, 0.8, 1.0])


def test_macro_f1_known():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    expected = np.mean([2.0 / 3.0, 0.8, 1.0])
    assert np.isclose(macro_f1(y_true, y_pred, k=3), expected)


def test_top_k_accuracy_known():
    # Row 0 true=2: top-2 prob classes are {0,2} -> hit.
    # Row 1 true=1: top-2 prob classes are {0,2} -> miss.
    proba = np.array(
        [
            [0.5, 0.1, 0.4],
            [0.6, 0.1, 0.3],
        ]
    )
    y_true = np.array([2, 1])
    assert top_k_accuracy(y_true, proba, k=2) == 0.5
    # k=1 reduces to plain top-1 accuracy: row0 pred 0 (miss), row1 pred 0 (miss).
    assert top_k_accuracy(y_true, proba, k=1) == 0.0


def test_empty_accuracy():
    assert accuracy(np.array([]), np.array([])) == 0.0
