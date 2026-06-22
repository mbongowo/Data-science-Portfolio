"""Known-answer tests for the ranking and rating-error metrics.

Every expected value here is *hand-computed* on a tiny example so that a green
test proves the metric is correct, not merely that it runs. The metrics module
has no third-party dependency beyond numpy, so these always execute.

Worked examples used below:

1. RMSE. y_true = [3, 5, 4], y_pred = [4, 5, 2].
   errors = [-1, 0, 2], squared = [1, 0, 4], mean = 5/3,
   RMSE = sqrt(5/3) = 1.2909944487358056.

2. Precision@K / Recall@K. recommended = [a, b, c, d, e],
   relevant = {b, d, x} (x is not recommended at all), k = 3.
   top-3 = [a, b, c]; hits = {b} => 1 hit.
   precision@3 = hits / k        = 1/3.
   recall@3    = hits / |relevant| = 1/3.

3. NDCG@K perfect ranking. relevance graded [3, 2, 1] presented in that
   (descending) order => DCG == IDCG => NDCG = 1.0.

4. NDCG@K non-trivial. recommended = [a, b, c, d], k = 4,
   relevance = {a: 3, b: 2, c: 3, d: 0}.
   discounts: rank 1 -> 1/log2(2)=1, rank 2 -> 1/log2(3), rank 3 -> 1/log2(4)=1/2,
              rank 4 -> 1/log2(5).
   DCG  = 3*1 + 2/log2(3) + 3*(1/2) + 0          = 5.7618595071429155
   ideal order of grades is [3, 3, 2, 0]:
   IDCG = 3*1 + 3/log2(3) + 2*(1/2) + 0          = 5.892789260714372
   NDCG = DCG / IDCG                              = 0.9777813616305049
"""

from __future__ import annotations

import math

import pytest

from recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k, rmse


def test_rmse_hand_value() -> None:
    """RMSE of [3,5,4] vs [4,5,2] is sqrt(5/3) (hand-derived)."""
    assert rmse([3.0, 5.0, 4.0], [4.0, 5.0, 2.0]) == pytest.approx(
        math.sqrt(5.0 / 3.0), abs=1e-12
    )


def test_rmse_perfect_is_zero() -> None:
    """Identical vectors give RMSE 0."""
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-12)


def test_rmse_rejects_bad_input() -> None:
    """Empty and mismatched inputs raise."""
    with pytest.raises(ValueError):
        rmse([], [])
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])


def test_precision_at_k_hand_value() -> None:
    """One hit in the top-3 => precision 1/3."""
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "x"}
    assert precision_at_k(recommended, relevant, 3) == pytest.approx(
        1.0 / 3.0, abs=1e-12
    )


def test_recall_at_k_hand_value() -> None:
    """One of three relevant items recalled in the top-3 => recall 1/3."""
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "x"}
    assert recall_at_k(recommended, relevant, 3) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_recall_with_no_relevant_is_zero() -> None:
    """No relevant items => recall is defined as 0."""
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_precision_recall_reject_bad_k() -> None:
    """k must be positive."""
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, -1)


def test_ndcg_perfect_ranking_is_one() -> None:
    """Items presented in descending relevance order => NDCG = 1.0."""
    recommended = ["a", "b", "c"]
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(recommended, relevance, 3) == pytest.approx(1.0, abs=1e-12)


def test_ndcg_hand_value() -> None:
    """Non-trivial ordering => NDCG = 0.9777813616305049 (see module docstring)."""
    recommended = ["a", "b", "c", "d"]
    relevance = {"a": 3.0, "b": 2.0, "c": 3.0, "d": 0.0}
    assert ndcg_at_k(recommended, relevance, 4) == pytest.approx(
        0.9777813616305049, abs=1e-12
    )


def test_ndcg_no_relevance_is_zero() -> None:
    """Nothing relevant => IDCG is zero => NDCG is defined as 0."""
    assert ndcg_at_k(["a", "b"], {}, 2) == 0.0


def test_ndcg_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1.0}, 0)
