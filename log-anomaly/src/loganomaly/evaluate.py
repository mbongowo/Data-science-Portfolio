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
    from numpy.typing import ArrayLike


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
