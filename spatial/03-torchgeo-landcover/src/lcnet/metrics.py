"""Pure-numpy multi-class classification metrics.

These functions depend only on :mod:`numpy`, so they import and test without
torch / torchvision / torchgeo installed. They operate on integer label vectors
``y_true`` / ``y_pred`` of class indices in ``range(num_classes)`` (whole-patch
land-cover labels), plus a probability matrix for the top-k metric.

Convention
----------
The confusion matrix has *true* labels along rows and *predicted* labels along
columns, so the diagonal holds the correctly classified samples. Per-class
precision / recall / F1 with an empty denominator (a class never predicted, or
absent from the truth) are reported as ``0.0`` rather than ``nan``: for a
macro-average over the fixed land-cover taxonomy a missed class should count as
a zero, not be silently dropped.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "confusion_matrix",
    "accuracy",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "macro_f1",
    "micro_f1",
    "cohen_kappa",
    "top_k_accuracy",
]


def _check_labels(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D int arrays and validate matching length."""
    t = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs y_pred {p.shape}")
    return t, p


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Dense ``num_classes x num_classes`` confusion matrix.

    Row ``i`` / column ``j`` counts samples whose *true* label is class ``i``
    and whose *predicted* label is class ``j``; the diagonal therefore holds the
    correct samples. Counting is vectorised with ``np.bincount`` on the
    flattened ``true * num_classes + pred`` index.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Integer label vectors of class indices in ``range(num_classes)``, of
        identical length.
    num_classes : int
        Number of classes; sets the matrix size.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(num_classes, num_classes)`` (true along rows,
        pred along columns).

    Raises
    ------
    ValueError
        If lengths differ, ``num_classes`` is not positive, or any label is
        outside ``range(num_classes)``.
    """
    t, p = _check_labels(y_true, y_pred)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    t = t.astype(np.int64)
    p = p.astype(np.int64)
    if t.size and (
        t.min() < 0 or t.max() >= num_classes or p.min() < 0 or p.max() >= num_classes
    ):
        raise ValueError("labels must lie in range(num_classes)")
    index = t * num_classes + p
    counts = np.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy: fraction of samples classified correctly.

    Returns ``1.0`` for an empty input (vacuously correct).
    """
    t, p = _check_labels(y_true, y_pred)
    if t.size == 0:
        return 1.0
    return float(np.mean(t == p))


def per_class_precision(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Per-class precision ``cm[k, k] / cm[:, k].sum()`` (correct / predicted-k).

    A class that is never predicted has a zero denominator and is reported as
    ``0.0``.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    diag = np.diag(cm)
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(predicted > 0, diag / predicted, 0.0)
    return out


def per_class_recall(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Per-class recall ``cm[k, k] / cm[k, :].sum()`` (correct / true-k).

    A class absent from the truth has a zero denominator and is reported as
    ``0.0``.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    diag = np.diag(cm)
    actual = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(actual > 0, diag / actual, 0.0)
    return out


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Per-class F1: harmonic mean of precision and recall.

    ``F1_k = 2 * P_k * R_k / (P_k + R_k)``; ``0.0`` where both are zero.
    """
    prec = per_class_precision(y_true, y_pred, num_classes)
    rec = per_class_recall(y_true, y_pred, num_classes)
    denom = prec + rec
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, 2.0 * prec * rec / denom, 0.0)
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1: unweighted mean of the per-class F1 scores.

    Every class counts equally regardless of support, so rare land-cover
    classes are not drowned out by common ones.
    """
    return float(np.mean(per_class_f1(y_true, y_pred, num_classes)))


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Micro-averaged F1: pool TP / FP / FN across all classes, then F1.

    For single-label multi-class problems the micro F1 equals the overall
    accuracy (every sample contributes exactly one true positive or one false
    positive *and* one false negative). It is provided so the README can show it
    diverging from macro F1 on an imbalanced case.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    tp = float(np.trace(cm))
    # Pooled FP and FN are equal for single-label assignment, but compute both
    # explicitly so the formula is transparent.
    fp = float(cm.sum(axis=0).sum() - tp)
    fn = float(cm.sum(axis=1).sum() - tp)
    denom = 2.0 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2.0 * tp) / denom


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Cohen's kappa: agreement corrected for chance.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement
    (accuracy) and ``p_e`` is the agreement expected if true and predicted
    labels were independent with the same marginals. ``1.0`` is perfect
    agreement and ``~0`` is chance-level.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    total = cm.sum()
    if total == 0:
        return 1.0
    observed = np.trace(cm) / total
    row_marginal = cm.sum(axis=1) / total
    col_marginal = cm.sum(axis=0) / total
    expected = float(np.sum(row_marginal * col_marginal))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Top-k accuracy: fraction of samples whose true class is in the top ``k``.

    Parameters
    ----------
    y_true : numpy.ndarray
        Integer true labels, shape ``(n_samples,)``.
    proba : numpy.ndarray
        Class-probability (or score) matrix, shape ``(n_samples, num_classes)``.
        Only the relative ordering of each row matters.
    k : int
        Number of top-scoring classes to admit. ``k=1`` reduces to ordinary
        accuracy.

    Returns
    -------
    float
        Top-k accuracy in ``[0, 1]``; ``1.0`` for an empty input.

    Raises
    ------
    ValueError
        If ``proba`` is not 2-D, its rows do not match ``len(y_true)``, or
        ``k`` is outside ``[1, num_classes]``.
    """
    t = np.asarray(y_true).reshape(-1)
    pr = np.asarray(proba, dtype=np.float64)
    if pr.ndim != 2:
        raise ValueError("proba must be a 2-D (n_samples, num_classes) array")
    if pr.shape[0] != t.shape[0]:
        raise ValueError(f"row mismatch: y_true {t.shape[0]} vs proba {pr.shape[0]}")
    num_classes = pr.shape[1]
    if not 1 <= k <= num_classes:
        raise ValueError("k must lie in [1, num_classes]")
    if t.size == 0:
        return 1.0
    # The k highest-scoring class indices per row (unordered within the top-k).
    topk = np.argpartition(-pr, kth=k - 1, axis=1)[:, :k]
    hit = np.any(topk == t[:, None], axis=1)
    return float(np.mean(hit))
