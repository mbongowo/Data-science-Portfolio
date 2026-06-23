"""Known-answer tests for the classification / regression metrics.

Every expected value is hand-computed on a tiny example. The metrics module has
no dependency beyond numpy, so these always run.

Worked classification example (used throughout):

    y_true = [1, 1, 1, 0, 0, 0, 0, 0]   (3 positives, 5 negatives)
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0]
    -> tp = 2 (rows 0,1), fp = 1 (row 3), fn = 1 (row 2), tn = 4.
    accuracy  = (tp + tn) / 8 = 6/8     = 0.75
    precision = tp / (tp + fp) = 2/3    = 0.666...
    recall    = tp / (tp + fn) = 2/3    = 0.666...
    f1        = 2PR/(P+R) = 2/3         = 0.666...  (P == R here)
"""

from __future__ import annotations

import math

import pytest

from mlpipe.metrics import (
    accuracy,
    confusion_counts,
    f1,
    mae,
    precision,
    r2,
    recall,
    rmse,
    roc_auc,
)

Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 0, 0]


def test_confusion_counts_hand_value() -> None:
    """tp=2, fp=1, fn=1, tn=4 on the worked example."""
    assert confusion_counts(Y_TRUE, Y_PRED) == (2, 1, 1, 4)


def test_accuracy_hand_value() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.75, abs=1e-12)


def test_precision_hand_value() -> None:
    assert precision(Y_TRUE, Y_PRED) == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_recall_hand_value() -> None:
    assert recall(Y_TRUE, Y_PRED) == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_f1_hand_value() -> None:
    assert f1(Y_TRUE, Y_PRED) == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_precision_no_positive_predictions_is_zero() -> None:
    """Nothing predicted positive => precision defined as 0."""
    assert precision([1, 0, 1], [0, 0, 0]) == 0.0


def test_recall_no_actual_positives_is_zero() -> None:
    """No positives in the labels => recall defined as 0."""
    assert recall([0, 0, 0], [0, 1, 0]) == 0.0


def test_f1_all_zero_is_zero() -> None:
    """Precision and recall both zero => F1 defined as 0 (no division error)."""
    assert f1([1, 1], [0, 0]) == 0.0


def test_roc_auc_perfect_separation_is_one() -> None:
    """Every positive scored above every negative => AUC = 1.0.

    y = [0, 1, 0, 1], scores = [0.1, 0.2, 0.8, 0.9]: positives at scores
    {0.2, 0.9}, negatives at {0.1, 0.8}. The pair (pos 0.2 vs neg 0.8) is
    mis-ordered, so 3 of 4 pairs are correct -> AUC = 0.75, *not* 1.0. To get a
    perfect separator the labels must line up with the score order:
    """
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]  # the two largest scores are the positives
    assert roc_auc(y, scores) == pytest.approx(1.0, abs=1e-12)


def test_roc_auc_all_tied_scores_is_half() -> None:
    """All scores equal (pure tie) => AUC = 0.5 by the average-rank convention."""
    y = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert roc_auc(y, scores) == pytest.approx(0.5, abs=1e-12)


def test_roc_auc_hand_value_with_one_misorder() -> None:
    """Hand-derived AUC with a single mis-ordered pair.

    y = [0, 1, 0, 1], scores = [0.1, 0.4, 0.35, 0.8].
    Positives are at scores {0.4, 0.8}, negatives at {0.1, 0.35}.
    Of the 2x2 = 4 positive/negative pairs, the only one where the negative is
    *not* below the positive is (pos 0.4 vs neg 0.35): 0.4 > 0.35 so that is
    still correct -> all 4 pairs correct -> AUC = 1.0. Swap to make 0.35 the
    positive instead: y = [0, 1, 1, 0] keeps scores; now pos = {0.4, 0.35},
    neg = {0.1, 0.8}; pair (0.4, 0.8) is wrong, (0.35, 0.8) is wrong, the other
    two correct -> 2/4 = 0.5.
    """
    assert roc_auc([0, 1, 1, 0], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(
        0.5, abs=1e-12
    )


def test_roc_auc_single_class_returns_half() -> None:
    """AUC is undefined with one class present; return 0.5."""
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.5


def test_rmse_hand_value() -> None:
    """RMSE of [3,5,4] vs [4,5,2] is sqrt(5/3)."""
    assert rmse([3.0, 5.0, 4.0], [4.0, 5.0, 2.0]) == pytest.approx(
        math.sqrt(5.0 / 3.0), abs=1e-12
    )


def test_mae_hand_value() -> None:
    """MAE of [3,5,4] vs [4,5,2] is (1 + 0 + 2)/3 = 1.0."""
    assert mae([3.0, 5.0, 4.0], [4.0, 5.0, 2.0]) == pytest.approx(1.0, abs=1e-12)


def test_r2_perfect_is_one() -> None:
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0, abs=1e-12)


def test_r2_mean_predictor_is_zero() -> None:
    """Predicting the mean of y gives R^2 = 0."""
    y = [1.0, 2.0, 3.0]
    assert r2(y, [2.0, 2.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


def test_r2_hand_value() -> None:
    """y=[1,2,3], pred=[1,2,2]: SS_res = 1, SS_tot = 2 => R^2 = 1 - 1/2 = 0.5."""
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 2.0]) == pytest.approx(0.5, abs=1e-12)


def test_metrics_reject_length_mismatch() -> None:
    with pytest.raises(ValueError):
        accuracy([1, 0], [1])
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
