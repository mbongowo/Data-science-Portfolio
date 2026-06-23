"""Classification and regression metrics in pure numpy.

A small, dependency-free metrics layer (numpy only) so the headline numbers are
checkable with hand-derived known-answer tests rather than trusted blindly from
a library.

Classification (binary, labels in ``{0, 1}``):

* :func:`confusion_counts` — the ``(tp, fp, fn, tn)`` quadruple every other
  classification metric is built from;
* :func:`accuracy`, :func:`precision`, :func:`recall`, :func:`f1`;
* :func:`roc_auc` — a rank-based AUC (the probability that a random positive is
  scored above a random negative), which handles ties correctly.

Regression:

* :func:`rmse`, :func:`mae`, :func:`r2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike


def _as_label_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple:
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    if a.size == 0:
        raise ValueError("Need at least one observation.")
    if a.shape != b.shape:
        raise ValueError(f"Length mismatch: {a.size} vs {b.size}.")
    return a, b


def confusion_counts(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` for binary labels in ``{0, 1}``.

    Parameters
    ----------
    y_true, y_pred:
        Equal-length arrays of 0/1 labels.

    Returns
    -------
    tuple of int
        True positives, false positives, false negatives, true negatives.
    """
    a, b = _as_label_arrays(y_true, y_pred)
    yt = a.astype(int)
    yp = b.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return tp, fp, fn, tn


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of predictions that match the label."""
    a, b = _as_label_arrays(y_true, y_pred)
    return float(np.mean(a.astype(int) == b.astype(int)))


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Precision ``tp / (tp + fp)``; ``0.0`` when nothing is predicted positive."""
    tp, fp, _fn, _tn = confusion_counts(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom else 0.0


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Recall ``tp / (tp + fn)``; ``0.0`` when there are no positives."""
    tp, _fp, fn, _tn = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom else 0.0


def f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Harmonic mean of precision and recall; ``0.0`` when both are zero."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return float(2.0 * p * r / denom) if denom else 0.0


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    r"""Rank-based ROC AUC for binary labels.

    AUC equals the probability that a randomly chosen positive is scored above a
    randomly chosen negative. Using the Mann-Whitney U identity with average
    ranks (so ties contribute 0.5):

    .. math::

        \mathrm{AUC} = \frac{R_+ - n_+(n_+ + 1)/2}{n_+ \, n_-},

    where ``R_+`` is the sum of the average ranks of the positive scores and
    ``n_+`` / ``n_-`` are the positive / negative counts.

    Parameters
    ----------
    y_true:
        0/1 labels.
    y_score:
        Real-valued scores (higher = more likely positive).

    Returns
    -------
    float
        AUC in ``[0, 1]``. A perfect separator gives 1.0; random scores tend to
        0.5. Returns 0.5 when one class is absent (AUC is undefined there).
    """
    yt, ys = _as_label_arrays(y_true, y_score)
    yt = yt.astype(int)
    ys = ys.astype(float)
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(ys, kind="mergesort")
    sorted_scores = ys[order]
    ranks = np.empty(ys.size, dtype=float)
    i = 0
    while i < ys.size:
        j = i
        while j + 1 < ys.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Average rank (1-based) for the tied block [i, j].
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = float(np.sum(ranks[yt == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Root-mean-square error :math:`\sqrt{\frac1n \sum (y_i - \hat y_i)^2}`."""
    a, b = _as_label_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a.astype(float) - b.astype(float)) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Mean absolute error :math:`\frac1n \sum |y_i - \hat y_i|`."""
    a, b = _as_label_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Coefficient of determination ``1 - SS_res / SS_tot``.

    ``1.0`` is a perfect fit; ``0.0`` is no better than predicting the mean; it
    can go negative for a fit worse than the mean. Returns ``0.0`` when the true
    values are constant (``SS_tot == 0``).
    """
    a, b = _as_label_arrays(y_true, y_pred)
    a = a.astype(float)
    b = b.astype(float)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot
