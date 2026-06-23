"""Pure-numpy classification metrics.

Every function here is hand-derivable from its definition and has no dependency
beyond numpy, so the test suite pins them against tiny known-answer examples.
Class indices are assumed to be integers in ``0..k-1``; pass ``k`` (the number
of classes) so a class absent from a particular split still gets its row/column.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of exactly-correct predictions."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """Return the ``(k, k)`` confusion matrix; rows = true, columns = predicted."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """Return the per-class F1 score as a length-``k`` array.

    For class ``c``: precision = TP / (TP + FP), recall = TP / (TP + FN), and
    F1 = 2 * precision * recall / (precision + recall). A class with no
    predictions and no truths (or a zero denominator) scores 0.
    """
    cm = confusion_matrix(y_true, y_pred, k)
    f1 = np.zeros(k, dtype=float)
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1[c] = 2.0 * precision * recall / (precision + recall)
    return f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Unweighted mean of the per-class F1 scores."""
    return float(np.mean(per_class_f1(y_true, y_pred, k)))


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Fraction of rows whose true class is among the top-``k`` probabilities.

    ``proba`` is ``(n_samples, n_classes)``; ``y_true`` holds the integer class
    index per row. With ``k = 1`` this equals plain accuracy.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be 2-D (n_samples, n_classes)")
    if y_true.shape[0] != proba.shape[0]:
        raise ValueError("y_true and proba must have the same number of rows")
    if y_true.size == 0:
        return 0.0
    k = int(min(k, proba.shape[1]))
    # Indices of the top-k classes per row (unordered is fine for membership).
    top = np.argsort(proba, axis=1)[:, -k:]
    hits = np.any(top == y_true[:, None], axis=1)
    return float(np.mean(hits))
