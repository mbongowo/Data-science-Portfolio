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

from loganomaly.evaluate import confusion_matrix, precision_recall_f1

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
