"""Known-answer tests for the pure-numpy metrics. Requires only numpy."""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.metrics import (
    cohen_kappa,
    confusion_counts,
    confusion_matrix,
    f1_score,
    frequency_weighted_iou,
    iou_score,
    mean_iou,
    mean_iou_multiclass,
    per_class_iou,
    per_class_precision,
    per_class_recall,
    pixel_accuracy,
    precision_score,
    recall_score,
)


def test_iou_perfect_match():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    assert iou_score(a, a) == 1.0


def test_iou_no_overlap():
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[0, 0], [1, 1]], dtype=bool)
    # intersection 0 / union 4 -> 0
    assert abs(iou_score(pred, target) - 0.0) < 1e-6


def test_iou_partial_known_answer():
    # pred has 3 ones, target has 3 ones, intersection = 2, union = 4
    pred = np.array([1, 1, 1, 0], dtype=bool)
    target = np.array([1, 1, 0, 1], dtype=bool)
    tp, fp, fn, tn = confusion_counts(pred, target)
    assert (tp, fp, fn, tn) == (2, 1, 1, 0)
    # IoU = 2 / (2 + 1 + 1) = 0.5
    assert abs(iou_score(pred, target) - 0.5) < 1e-6


def test_f1_partial_known_answer():
    pred = np.array([1, 1, 1, 0], dtype=bool)
    target = np.array([1, 1, 0, 1], dtype=bool)
    # F1 = 2*2 / (2*2 + 1 + 1) = 4/6 = 0.6667
    assert abs(f1_score(pred, target) - (4.0 / 6.0)) < 1e-6


def test_both_empty_is_one():
    z = np.zeros((3, 3), dtype=bool)
    assert iou_score(z, z) == 1.0
    assert f1_score(z, z) == 1.0


def test_threshold_on_probabilities():
    prob = np.array([0.9, 0.1, 0.6, 0.4])
    target = np.array([1, 0, 1, 0], dtype=bool)
    # thresholded pred == target -> perfect
    assert abs(iou_score(prob, target, threshold=0.5) - 1.0) < 1e-6


def test_mean_iou_batch():
    preds = np.array(
        [
            [[1, 1], [0, 0]],
            [[1, 0], [1, 0]],
        ],
        dtype=bool,
    )
    targets = np.array(
        [
            [[1, 1], [0, 0]],  # perfect -> 1.0
            [[0, 0], [1, 1]],  # inter 1, union 3 -> 1/3
        ],
        dtype=bool,
    )
    expected = (1.0 + (1.0 / 3.0)) / 2.0
    assert abs(mean_iou(preds, targets) - expected) < 1e-6


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        iou_score(np.zeros((2, 2)), np.zeros((3, 3)))


# --- precision / recall ---------------------------------------------------


def test_precision_recall_known_answer():
    pred = np.array([1, 1, 1, 0], dtype=bool)
    target = np.array([1, 1, 0, 1], dtype=bool)
    # tp=2, fp=1, fn=1 -> precision 2/3, recall 2/3
    assert abs(precision_score(pred, target) - (2.0 / 3.0)) < 1e-6
    assert abs(recall_score(pred, target) - (2.0 / 3.0)) < 1e-6


def test_precision_recall_asymmetric():
    # over-predicting: precision falls, recall stays high
    pred = np.array([1, 1, 1, 1], dtype=bool)
    target = np.array([1, 1, 0, 0], dtype=bool)
    # tp=2, fp=2, fn=0 -> precision 0.5, recall 1.0
    assert abs(precision_score(pred, target) - 0.5) < 1e-6
    assert abs(recall_score(pred, target) - 1.0) < 1e-6


def test_precision_one_when_nothing_predicted():
    # no positive prediction -> no false alarms -> precision 1.0 by convention
    pred = np.zeros(4, dtype=bool)
    target = np.array([1, 0, 0, 0], dtype=bool)
    assert precision_score(pred, target) == 1.0
    # but recall is 0: the one positive was missed
    assert abs(recall_score(pred, target) - 0.0) < 1e-6


def test_recall_one_when_target_empty():
    # empty target -> nothing to miss -> recall 1.0 by convention
    pred = np.array([1, 0, 0, 0], dtype=bool)
    target = np.zeros(4, dtype=bool)
    assert recall_score(pred, target) == 1.0


# --- ignore_index (binary) ------------------------------------------------


def test_ignore_index_drops_void_pixels():
    # last pixel is void (255) and must not be counted
    pred = np.array([1, 1, 1, 0, 1], dtype=int)
    target = np.array([1, 1, 0, 1, 255], dtype=int)
    # after dropping index 4: pred=[1,1,1,0] target=[1,1,0,1]
    assert confusion_counts(pred, target, ignore_index=255) == (2, 1, 1, 0)
    assert abs(iou_score(pred, target, ignore_index=255) - 0.5) < 1e-6


def test_ignore_index_all_void_is_one():
    # every pixel ignored -> both masks empty after masking -> 1.0
    pred = np.array([1, 0, 1], dtype=int)
    target = np.array([255, 255, 255], dtype=int)
    assert iou_score(pred, target, ignore_index=255) == 1.0
    assert f1_score(pred, target, ignore_index=255) == 1.0


def test_ignore_index_changes_result():
    pred = np.array([1, 1, 0], dtype=int)
    target = np.array([1, 0, 0], dtype=int)
    without = iou_score(pred, target)
    # marking the false-positive pixel as void removes the fp
    target_void = np.array([1, 255, 0], dtype=int)
    with_void = iou_score(pred, target_void, ignore_index=255)
    assert without < with_void
    assert with_void == 1.0


# --- multi-class IoU ------------------------------------------------------


def test_per_class_iou_known_answer():
    pred = np.array([[0, 0, 1], [2, 2, 1]])
    target = np.array([[0, 0, 1], [2, 1, 1]])
    # class 0: inter 2, union 2 -> 1.0
    # class 1: inter 2, union 3 -> 2/3
    # class 2: inter 1, union 2 -> 0.5
    ious = per_class_iou(pred, target, num_classes=3)
    assert abs(ious[0] - 1.0) < 1e-6
    assert abs(ious[1] - (2.0 / 3.0)) < 1e-6
    assert abs(ious[2] - 0.5) < 1e-6


def test_mean_iou_multiclass_macro_average():
    pred = np.array([[0, 0, 1], [2, 2, 1]])
    target = np.array([[0, 0, 1], [2, 1, 1]])
    expected = (1.0 + (2.0 / 3.0) + 0.5) / 3.0
    assert abs(mean_iou_multiclass(pred, target, num_classes=3) - expected) < 1e-6


def test_per_class_iou_absent_class_is_nan():
    # classes 2 and 3 never appear -> nan, not 0
    pred = np.array([0, 0, 1])
    target = np.array([0, 0, 1])
    ious = per_class_iou(pred, target, num_classes=4)
    assert np.isnan(ious[2]) and np.isnan(ious[3])
    # macro average skips the nan classes
    assert abs(mean_iou_multiclass(pred, target, num_classes=4) - 1.0) < 1e-6


def test_multiclass_ignore_index():
    pred = np.array([0, 1, 2, 1])
    target = np.array([0, 1, 2, 255])
    # last pixel void; remaining is a perfect match on classes 0,1,2
    assert abs(mean_iou_multiclass(pred, target, 3, ignore_index=255) - 1.0) < 1e-6


def test_multiclass_all_void_is_one():
    pred = np.array([0, 1])
    target = np.array([255, 255])
    assert mean_iou_multiclass(pred, target, 3, ignore_index=255) == 1.0


def test_per_class_iou_bad_num_classes_raises():
    with pytest.raises(ValueError):
        per_class_iou(np.zeros(4, dtype=int), np.zeros(4, dtype=int), num_classes=0)


# --- confusion matrix -----------------------------------------------------


def test_confusion_matrix_known_answer():
    # target along rows, pred along columns.
    target = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(pred, target, num_classes=3)
    expected = np.array(
        [
            [1, 1, 0],  # true 0: one predicted 0, one predicted 1
            [0, 2, 0],  # true 1: both predicted 1
            [1, 0, 0],  # true 2: predicted 0
        ]
    )
    assert np.array_equal(cm, expected)
    # rows sum to the per-class true counts
    assert np.array_equal(cm.sum(axis=1), [2, 2, 1])


def test_confusion_matrix_perfect_is_diagonal():
    a = np.array([[0, 1, 2], [2, 1, 0]])
    cm = confusion_matrix(a, a, num_classes=3)
    assert np.array_equal(cm, np.diag([2, 2, 2]))


def test_confusion_matrix_ignore_index():
    pred = np.array([0, 1, 2, 1])
    target = np.array([0, 1, 2, 255])
    cm = confusion_matrix(pred, target, num_classes=3, ignore_index=255)
    assert np.array_equal(cm, np.diag([1, 1, 1]))


def test_confusion_matrix_bad_num_classes_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros(3, dtype=int), np.zeros(3, dtype=int), 0)


# --- pixel accuracy -------------------------------------------------------


def test_pixel_accuracy_known_answer():
    pred = np.array([0, 1, 1, 2])
    target = np.array([0, 1, 2, 2])
    # 3 of 4 correct
    assert abs(pixel_accuracy(pred, target) - 0.75) < 1e-9


def test_pixel_accuracy_ignore_index():
    pred = np.array([0, 1, 1, 2])
    target = np.array([0, 1, 2, 255])
    # drop the void pixel: pred=[0,1,1] target=[0,1,2] -> 2/3 correct
    assert abs(pixel_accuracy(pred, target, ignore_index=255) - (2.0 / 3.0)) < 1e-9


def test_pixel_accuracy_all_void_is_one():
    pred = np.array([0, 1])
    target = np.array([255, 255])
    assert pixel_accuracy(pred, target, ignore_index=255) == 1.0


# --- per-class precision / recall from the confusion matrix ---------------


def test_per_class_precision_recall_from_cm():
    target = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(pred, target, num_classes=3)
    # precision = diag / column sum: col0=[1,0,1]->den2 diag1=0.5;
    #   col1=[1,2,0]->den3 diag2=2/3; col2 never predicted -> nan
    prec = per_class_precision(cm)
    assert abs(prec[0] - 0.5) < 1e-9
    assert abs(prec[1] - (2.0 / 3.0)) < 1e-9
    assert np.isnan(prec[2])
    # recall = diag / row sum: row0 den2 diag1=0.5; row1 den2 diag2=1.0;
    #   row2 den1 diag0=0.0
    rec = per_class_recall(cm)
    assert abs(rec[0] - 0.5) < 1e-9
    assert abs(rec[1] - 1.0) < 1e-9
    assert abs(rec[2] - 0.0) < 1e-9


# --- frequency-weighted IoU -----------------------------------------------


def test_frequency_weighted_iou_known_answer():
    # 4 pixels: classes 0,0,1,1. pred 0,1,1,1.
    pred = np.array([0, 1, 1, 1])
    target = np.array([0, 0, 1, 1])
    # class 0: inter1 union(2+1-1=2) -> 0.5 ; freq 2/4=0.5
    # class 1: inter2 union(2+3-2=3) -> 2/3 ; freq 2/4=0.5
    # fwiou = 0.5*0.5 + 0.5*(2/3) = 0.25 + 1/3 = 0.583333...
    expected = 0.5 * 0.5 + 0.5 * (2.0 / 3.0)
    assert abs(frequency_weighted_iou(pred, target, 2) - expected) < 1e-9


def test_frequency_weighted_iou_perfect_is_one():
    a = np.array([0, 0, 1, 2, 2])
    assert abs(frequency_weighted_iou(a, a, 3) - 1.0) < 1e-9


def test_frequency_weighted_iou_all_void_is_one():
    pred = np.array([0, 1])
    target = np.array([255, 255])
    assert frequency_weighted_iou(pred, target, 3, ignore_index=255) == 1.0


# --- Cohen's kappa --------------------------------------------------------


def test_cohen_kappa_perfect_agreement_is_one():
    a = np.array([0, 1, 2, 0, 1, 2])
    assert abs(cohen_kappa(a, a, 3) - 1.0) < 1e-9


def test_cohen_kappa_independent_is_near_zero():
    # Independent labels with matched marginals -> kappa ~ 0.
    rng = np.random.default_rng(0)
    n = 20000
    target = rng.integers(0, 3, size=n)
    pred = rng.integers(0, 3, size=n)
    k = cohen_kappa(pred, target, 3)
    assert abs(k) < 0.05


def test_cohen_kappa_known_two_class():
    # Hand-built 2x2: po and pe computed by hand.
    # target rows, pred cols:
    #   true0: pred0=8, pred1=2  (row sum 10)
    #   true1: pred0=3, pred1=7  (row sum 10)
    target = np.array([0] * 10 + [1] * 10)
    pred = np.array([0] * 8 + [1] * 2 + [0] * 3 + [1] * 7)
    # po = (8+7)/20 = 0.75
    # marginals: rows 0.5/0.5 ; cols (11/20, 9/20)=(0.55,0.45)
    # pe = 0.5*0.55 + 0.5*0.45 = 0.5
    # kappa = (0.75-0.5)/(1-0.5) = 0.5
    assert abs(cohen_kappa(pred, target, 2) - 0.5) < 1e-9


def test_cohen_kappa_single_class_is_one_when_perfect():
    # only class 0 present and predicted perfectly -> pe == 1, defined as 1.0
    a = np.zeros(5, dtype=int)
    assert cohen_kappa(a, a, 3) == 1.0
