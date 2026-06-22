"""Known-answer tests for the classification metrics.

Hand-worked example (positive class = anomaly = True):

    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 1, 1, 0, 0]

    position 0: TP    position 1: TP    position 2: FP
    position 3: TN    position 4: FN

    => tp = 2, fp = 1, fn = 1, tn = 1
       precision = 2 / 3, recall = 2 / 3, F1 = 2 / 3
"""

from __future__ import annotations

import numpy as np
import pytest

from loganomaly.evaluate import (
    auc,
    confusion_matrix,
    pr_curve,
    precision_recall_f1,
    roc_curve,
)

Y_TRUE = np.array([1, 1, 0, 0, 1], dtype=bool)
Y_PRED = np.array([1, 1, 1, 0, 0], dtype=bool)


def test_confusion_matrix_known() -> None:
    """The worked example gives (tn, fp, fn, tp) = (1, 1, 1, 2)."""
    assert confusion_matrix(Y_TRUE, Y_PRED) == (1, 1, 1, 2)


def test_precision_recall_f1_known() -> None:
    """P = R = F1 = 2/3 for the worked example."""
    p, r, f1 = precision_recall_f1(Y_TRUE, Y_PRED)
    assert p == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert r == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert f1 == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_perfect_prediction() -> None:
    """Identical truth and prediction => P = R = F1 = 1."""
    p, r, f1 = precision_recall_f1(Y_TRUE, Y_TRUE)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_no_predicted_positives_gives_zero() -> None:
    """No predicted positives => precision and recall and F1 are 0."""
    p, r, f1 = precision_recall_f1(Y_TRUE, np.zeros_like(Y_TRUE))
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_mismatched_lengths_raise() -> None:
    """Different-length inputs raise."""
    with pytest.raises(ValueError):
        confusion_matrix(np.array([True, False]), np.array([True]))


# --- Curves and AUC ---------------------------------------------------------
#
# Hand-worked separated case (positives all score above negatives):
#
#     y_true = [1, 1, 0, 0]
#     scores = [0.9, 0.8, 0.2, 0.1]
#
# Sweeping the threshold down the sorted scores:
#   t>=0.9 : tp=1 fp=0  -> tpr 0.5  fpr 0.0
#   t>=0.8 : tp=2 fp=0  -> tpr 1.0  fpr 0.0   (corner (0,1))
#   t>=0.2 : tp=2 fp=1  -> tpr 1.0  fpr 0.5
#   t>=0.1 : tp=2 fp=2  -> tpr 1.0  fpr 1.0
# The ROC hits (0,1), so the area under it is exactly 1.0.

SEP_TRUE = np.array([1, 1, 0, 0], dtype=bool)
SEP_SCORES = np.array([0.9, 0.8, 0.2, 0.1])


def test_roc_curve_perfect_separation_known() -> None:
    """Separated scores give the exact ROC vertices and pass through (0, 1)."""
    fpr, tpr = roc_curve(SEP_TRUE, SEP_SCORES)
    assert fpr.tolist() == [0.0, 0.0, 0.0, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 1.0, 1.0, 1.0]


def test_roc_auc_perfect_separation_is_one() -> None:
    """Known answer required by the spec: perfect separation => ROC-AUC == 1.0."""
    fpr, tpr = roc_curve(SEP_TRUE, SEP_SCORES)
    assert auc(fpr, tpr) == pytest.approx(1.0, abs=1e-12)


def test_roc_auc_perfect_inversion_is_zero() -> None:
    """If every positive scores below every negative, ROC-AUC is 0."""
    fpr, tpr = roc_curve(SEP_TRUE, -SEP_SCORES)
    assert auc(fpr, tpr) == pytest.approx(0.0, abs=1e-12)


def test_roc_auc_random_tie_is_half() -> None:
    """All-equal scores: the ROC is the diagonal, AUC = 0.5."""
    fpr, tpr = roc_curve(np.array([1, 0, 1, 0], dtype=bool), np.ones(4))
    assert auc(fpr, tpr) == pytest.approx(0.5, abs=1e-12)


def test_pr_curve_perfect_separation() -> None:
    """Separated scores: precision stays 1.0 until all positives are recovered."""
    recall, precision = pr_curve(SEP_TRUE, SEP_SCORES)
    # recall 0 (pinned), 0.5, 1.0 at precision 1.0, then precision falls.
    assert recall.tolist() == [0.0, 0.5, 1.0, 1.0, 1.0]
    assert precision[:3].tolist() == [1.0, 1.0, 1.0]
    # Average precision (area under PR) is 1.0 for perfect separation.
    assert auc(recall, precision) == pytest.approx(1.0, abs=1e-12)


def test_roc_single_class_raises() -> None:
    """ROC is undefined with only one class present."""
    with pytest.raises(ValueError):
        roc_curve(np.array([1, 1, 1], dtype=bool), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        roc_curve(np.array([0, 0, 0], dtype=bool), np.array([0.1, 0.2, 0.3]))


def test_pr_no_positives_raises() -> None:
    """PR curve needs at least one positive."""
    with pytest.raises(ValueError):
        pr_curve(np.array([0, 0, 0], dtype=bool), np.array([0.1, 0.2, 0.3]))


def test_auc_unsorted_x_orientation_independent() -> None:
    """auc sorts x first, so shuffled inputs give the same area."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])  # triangle, area 1.0
    assert auc(x, y) == pytest.approx(1.0, abs=1e-12)
    perm = [2, 0, 1]
    assert auc(x[perm], y[perm]) == pytest.approx(1.0, abs=1e-12)


def test_auc_needs_two_points() -> None:
    """A single point cannot be integrated."""
    with pytest.raises(ValueError):
        auc(np.array([0.0]), np.array([1.0]))
