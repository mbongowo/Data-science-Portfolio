"""Binary classification metrics for scored sessions (pure python / numpy).

The detectors emit a boolean flag per session (anomaly or not). When labels are
available — Loghub HDFS_v1 ships per-block Normal / Anomaly labels — these
functions score the flags against the truth. The positive class is "anomaly"
(``True``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for boolean truth and prediction vectors.

    The positive class is ``True`` (anomaly). With

    * ``y_true = [1, 1, 0, 0, 1]`` and
    * ``y_pred = [1, 1, 1, 0, 0]``

    the counts are ``tp = 2`` (positions 0, 1), ``fp = 1`` (position 2),
    ``fn = 1`` (position 4), ``tn = 1`` (position 3).

    Parameters
    ----------
    y_true:
        Ground-truth labels, coerced to bool.
    y_pred:
        Predicted labels, coerced to bool.

    Returns
    -------
    tuple[int, int, int, int]
        ``(tn, fp, fn, tp)``.

    Raises
    ------
    ValueError
        If the inputs have different lengths.
    """
    t = np.asarray(y_true).astype(bool).ravel()
    p = np.asarray(y_pred).astype(bool).ravel()
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    tp = int(np.sum(t & p))
    tn = int(np.sum(~t & ~p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    return tn, fp, fn, tp


def precision_recall_f1(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` for boolean truth and prediction.

    With ``tp = 2``, ``fp = 1``, ``fn = 1`` (the worked example in
    :func:`confusion_matrix`):

    * precision = tp / (tp + fp) = 2 / 3,
    * recall    = tp / (tp + fn) = 2 / 3,
    * F1        = 2PR / (P + R)  = 2 / 3.

    A metric whose denominator is zero is defined as ``0.0`` (no predicted
    positives => precision 0; no actual positives => recall 0; both zero => F1 0).

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.

    Returns
    -------
    tuple[float, float, float]
        ``(precision, recall, f1)``.
    """
    _, fp, fn, tp = confusion_matrix(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _threshold_sweep(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Sort by descending score; return aligned truth and the unique cut scores.

    A threshold-sweep detector flags every session with ``score >= t``. Sweeping
    ``t`` from high to low adds one (or, for ties, several) sessions at a time.
    The natural set of thresholds is therefore the distinct score values seen.
    """
    t = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if t.shape != s.shape:
        raise ValueError("y_true and scores must have the same length.")
    if t.size == 0:
        raise ValueError("inputs must be non-empty.")

    order = np.argsort(-s, kind="stable")
    return t[order], s[order]


def roc_curve(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ROC points ``(fpr, tpr)`` swept over score thresholds, high to low.

    The positive class is ``True``. As the threshold drops from ``+inf`` to
    ``-inf``, more sessions are flagged; the curve starts at ``(0, 0)`` (flag
    nothing) and ends at ``(1, 1)`` (flag everything). Ties in the score are
    resolved together so the curve is well defined when scores repeat.

    A perfectly separated set of scores (every positive scored above every
    negative) passes through ``(0, 1)`` and gives :func:`auc` == ``1.0``.

    Parameters
    ----------
    y_true:
        Ground-truth labels, coerced to bool.
    scores:
        Per-session anomaly scores; higher means more anomalous.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(fpr, tpr)`` arrays of equal length, both starting at 0 and ending
        at 1, ordered by increasing ``fpr``.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or contain only one class.
    """
    t, s = _threshold_sweep(y_true, scores)
    n_pos = int(t.sum())
    n_neg = int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both a positive and a negative class.")

    # Cumulative TP / FP as the threshold sweeps down the sorted scores.
    tps = np.cumsum(t.astype(float))
    fps = np.cumsum((~t).astype(float))

    # Keep only the last index of each tied score group (a threshold sits
    # *between* distinct scores, so tied rows must be flagged together).
    distinct = np.r_[np.diff(s) != 0, True]
    tps = np.r_[0.0, tps[distinct]]
    fps = np.r_[0.0, fps[distinct]]

    tpr = tps / n_pos
    fpr = fps / n_neg
    return fpr, tpr


def pr_curve(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precision-recall points ``(recall, precision)`` over score thresholds.

    The positive class is ``True``. Points are ordered by increasing recall, as
    the threshold drops and more sessions are flagged. The first point has
    recall ``0``; its precision is pinned to that of the first non-empty flag set
    so the curve has a defined left endpoint. Tied scores are grouped, matching
    :func:`roc_curve`.

    Parameters
    ----------
    y_true:
        Ground-truth labels, coerced to bool.
    scores:
        Per-session anomaly scores; higher means more anomalous.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(recall, precision)`` arrays of equal length, ordered by increasing
        recall, ending at recall ``1.0``.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or contain no positives.
    """
    t, s = _threshold_sweep(y_true, scores)
    n_pos = int(t.sum())
    if n_pos == 0:
        raise ValueError("PR curve needs at least one positive.")

    tps = np.cumsum(t.astype(float))
    fps = np.cumsum((~t).astype(float))

    distinct = np.r_[np.diff(s) != 0, True]
    tps = tps[distinct]
    fps = fps[distinct]

    recall = tps / n_pos
    precision = tps / (tps + fps)

    # Left endpoint at recall 0, precision = that of the first flagged group.
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]
    return recall, precision


def auc(x: ArrayLike, y: ArrayLike) -> float:
    """Area under the curve ``y(x)`` by the trapezoid rule.

    Integrates ``y`` against ``x`` with :func:`numpy.trapezoid`. ``x`` need not
    be sorted on input — it is sorted ascending first (carrying ``y`` with it) so
    the area is orientation-independent and always non-negative for a monotone
    curve. A perfectly separated ROC curve gives ``1.0``.

    Parameters
    ----------
    x:
        Horizontal coordinates (e.g. FPR, or recall).
    y:
        Vertical coordinates (e.g. TPR, or precision).

    Returns
    -------
    float
        The signed area under the sorted ``(x, y)`` curve.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length or have fewer than two points.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same length.")
    if xs.size < 2:
        raise ValueError("need at least two points to integrate.")

    order = np.argsort(xs, kind="stable")
    return float(np.trapezoid(ys[order], xs[order]))
