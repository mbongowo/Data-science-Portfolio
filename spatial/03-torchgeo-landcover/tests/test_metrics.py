"""Known-answer tests for the pure-numpy classification metrics. numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from lcnet.metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    micro_f1,
    per_class_f1,
    per_class_precision,
    per_class_recall,
    top_k_accuracy,
)


def test_confusion_matrix_known_answer():
    # true along rows, pred along columns.
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    expected = np.array(
        [
            [1, 1, 0],  # true 0: one pred 0, one pred 1
            [0, 2, 0],  # true 1: both pred 1
            [1, 0, 0],  # true 2: pred 0
        ]
    )
    assert np.array_equal(cm, expected)
    assert np.array_equal(cm.sum(axis=1), [2, 2, 1])


def test_confusion_matrix_perfect_is_diagonal():
    a = np.array([0, 1, 2, 2, 1, 0])
    cm = confusion_matrix(a, a, num_classes=3)
    assert np.array_equal(cm, np.diag([2, 2, 2]))


def test_confusion_matrix_bad_num_classes_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros(3, dtype=int), np.zeros(3, dtype=int), 0)


def test_confusion_matrix_label_out_of_range_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), num_classes=3)


def test_accuracy_known_answer():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    assert abs(accuracy(y_true, y_pred) - 0.75) < 1e-9


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0, 1, 2]))


def test_per_class_precision_recall_f1_known_answer():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    # precision = diag / col sum:
    #   col0 = [1,0,1] den2 diag1 -> 0.5
    #   col1 = [1,2,0] den3 diag2 -> 2/3
    #   col2 never predicted -> 0.0 by convention
    prec = per_class_precision(y_true, y_pred, 3)
    assert abs(prec[0] - 0.5) < 1e-9
    assert abs(prec[1] - (2.0 / 3.0)) < 1e-9
    assert prec[2] == 0.0
    # recall = diag / row sum:
    #   row0 den2 diag1 -> 0.5 ; row1 den2 diag2 -> 1.0 ; row2 den1 diag0 -> 0.0
    rec = per_class_recall(y_true, y_pred, 3)
    assert abs(rec[0] - 0.5) < 1e-9
    assert abs(rec[1] - 1.0) < 1e-9
    assert rec[2] == 0.0
    # f1_0 = 2*0.5*0.5/(0.5+0.5) = 0.5
    # f1_1 = 2*(2/3)*1/((2/3)+1) = (4/3)/(5/3) = 0.8
    f1 = per_class_f1(y_true, y_pred, 3)
    assert abs(f1[0] - 0.5) < 1e-9
    assert abs(f1[1] - 0.8) < 1e-9
    assert f1[2] == 0.0


def test_macro_vs_micro_f1_differ_on_imbalanced():
    # Heavily imbalanced: class 0 dominant and perfect; rare class 1 partly wrong.
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    # micro f1 == accuracy == 9/10 = 0.9
    assert abs(micro_f1(y_true, y_pred, 2) - 0.9) < 1e-9
    # macro f1 averages the two classes' f1 and is dragged down by the rare one.
    macro = macro_f1(y_true, y_pred, 2)
    assert macro < micro_f1(y_true, y_pred, 2)
    # class1: precision 1.0, recall 0.5 -> f1 = 2/3; class0 f1 ~ 0.9412
    # macro = (0.94117... + 0.66667) / 2 = 0.80392...
    assert abs(macro - ((16.0 / 17.0) + (2.0 / 3.0)) / 2.0) < 1e-9


def test_cohen_kappa_perfect_is_one():
    a = np.array([0, 1, 2, 0, 1, 2])
    assert abs(cohen_kappa(a, a, 3) - 1.0) < 1e-9


def test_cohen_kappa_independent_is_near_zero():
    rng = np.random.default_rng(0)
    n = 20000
    y_true = rng.integers(0, 3, size=n)
    y_pred = rng.integers(0, 3, size=n)
    assert abs(cohen_kappa(y_true, y_pred, 3)) < 0.05


def test_cohen_kappa_known_two_class():
    # true rows, pred cols: true0 -> pred0=8,pred1=2 ; true1 -> pred0=3,pred1=7
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 8 + [1] * 2 + [0] * 3 + [1] * 7)
    # po = 15/20 = 0.75 ; pe = 0.5*0.55 + 0.5*0.45 = 0.5 ; kappa = 0.5
    assert abs(cohen_kappa(y_true, y_pred, 2) - 0.5) < 1e-9


def test_top_k_accuracy_known_answer():
    y_true = np.array([0, 1, 2])
    # row0: true class 0 is the argmax -> in top-1 and top-2
    # row1: true class 1 is 2nd highest -> in top-2, NOT top-1
    # row2: true class 2 is lowest -> in neither
    proba = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1],
            [0.5, 0.4, 0.1],
        ]
    )
    assert abs(top_k_accuracy(y_true, proba, k=1) - (1.0 / 3.0)) < 1e-9
    assert abs(top_k_accuracy(y_true, proba, k=2) - (2.0 / 3.0)) < 1e-9
    assert top_k_accuracy(y_true, proba, k=3) == 1.0


def test_top_k_accuracy_bad_k_raises():
    with pytest.raises(ValueError):
        top_k_accuracy(np.array([0, 1]), np.zeros((2, 3)), k=0)
    with pytest.raises(ValueError):
        top_k_accuracy(np.array([0, 1]), np.zeros((2, 3)), k=4)


def test_top_k_accuracy_shape_guard():
    with pytest.raises(ValueError):
        top_k_accuracy(np.array([0, 1]), np.zeros(3), k=1)  # not 2-D
    with pytest.raises(ValueError):
        top_k_accuracy(np.array([0, 1, 2]), np.zeros((2, 3)), k=1)  # row mismatch
