"""Known-answer tests for the pure-numpy metrics. Requires only numpy."""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.metrics import (
    confusion_counts,
    f1_score,
    iou_score,
    mean_iou,
    mean_iou_multiclass,
    per_class_iou,
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
