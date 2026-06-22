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

5. Average Precision@K. recommended = [a, b, c, d, e],
   relevant = {b, d, x}, k = 4. Top-4 = [a, b, c, d].
   rank 1 (a): miss.  rank 2 (b): hit #1 -> precision 1/2.
   rank 3 (c): miss.  rank 4 (d): hit #2 -> precision 2/4 = 1/2.
   sum of hit-precisions = 1/2 + 1/2 = 1.
   denominator = min(|relevant|, k) = min(3, 4) = 3.
   AP@4 = 1 / 3 = 0.3333333333333333.

6. Mean Reciprocal Rank. lists = [[a, b, c], [x, y, z]],
   relevant = [{b}, {x}]. user 1: first relevant (b) at rank 2 -> 1/2.
   user 2: first relevant (x) at rank 1 -> 1/1 = 1.
   MRR = (1/2 + 1) / 2 = 0.75.

7. Catalogue coverage. lists = [[a, b], [b, c]], catalog = {a, b, c, d}.
   union of recommended-and-in-catalog = {a, b, c} -> 3 of 4 items.
   coverage = 3 / 4 = 0.75.
"""

from __future__ import annotations

import math

import pytest

from recsys.metrics import (
    average_precision_at_k,
    catalog_coverage,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


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


def test_average_precision_hand_value() -> None:
    """Hits at ranks 2 and 4 => AP@4 = (1/2 + 1/2)/3 = 1/3 (see docstring)."""
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "x"}
    assert average_precision_at_k(recommended, relevant, 4) == pytest.approx(
        1.0 / 3.0, abs=1e-12
    )


def test_average_precision_perfect_is_one() -> None:
    """All relevant items at the very top => AP = 1.0."""
    assert average_precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(
        1.0, abs=1e-12
    )


def test_average_precision_no_relevant_is_zero() -> None:
    """No relevant items => AP defined as 0."""
    assert average_precision_at_k(["a", "b"], set(), 2) == 0.0


def test_average_precision_no_hits_is_zero() -> None:
    """Relevant items exist but none are recommended => AP = 0."""
    assert average_precision_at_k(["a", "b"], {"x", "y"}, 2) == 0.0


def test_average_precision_k_larger_than_catalog() -> None:
    """k beyond the list length scores only what is present (no index error)."""
    # top-10 of a 2-item list: only b is relevant, hit at rank 2 -> 1/2;
    # denom = min(|relevant|=1, k=10) = 1 -> AP = 1/2.
    assert average_precision_at_k(["a", "b"], {"b"}, 10) == pytest.approx(
        0.5, abs=1e-12
    )


def test_average_precision_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        average_precision_at_k(["a"], {"a"}, 0)


def test_mrr_hand_value() -> None:
    """First-relevant at ranks 2 and 1 => MRR = (1/2 + 1)/2 = 0.75."""
    lists = [["a", "b", "c"], ["x", "y", "z"]]
    relevant = [{"b"}, {"x"}]
    assert mean_reciprocal_rank(lists, relevant) == pytest.approx(0.75, abs=1e-12)


def test_mrr_no_relevant_anywhere_is_zero() -> None:
    """A user with no relevant item in the list contributes 0 reciprocal rank."""
    lists = [["a", "b"], ["c", "d"]]
    relevant = [{"z"}, {"q"}]
    assert mean_reciprocal_rank(lists, relevant) == 0.0


def test_mrr_empty_is_zero() -> None:
    """No users => MRR = 0."""
    assert mean_reciprocal_rank([], []) == 0.0


def test_mrr_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


def test_catalog_coverage_hand_value() -> None:
    """Recommended union {a,b,c} of catalog {a,b,c,d} => coverage 3/4 = 0.75."""
    lists = [["a", "b"], ["b", "c"]]
    catalog = {"a", "b", "c", "d"}
    assert catalog_coverage(lists, catalog) == pytest.approx(0.75, abs=1e-12)


def test_catalog_coverage_full() -> None:
    """Every catalogue item recommended somewhere => coverage 1.0."""
    lists = [["a", "b"], ["c"]]
    assert catalog_coverage(lists, {"a", "b", "c"}) == pytest.approx(1.0, abs=1e-12)


def test_catalog_coverage_empty_recommendations_is_zero() -> None:
    """No recommendations at all => nothing covered."""
    assert catalog_coverage([], {"a", "b"}) == 0.0
    assert catalog_coverage([[], []], {"a", "b"}) == 0.0


def test_catalog_coverage_empty_catalog_is_zero() -> None:
    """Empty catalogue => coverage defined as 0 (no division by zero)."""
    assert catalog_coverage([["a"]], set()) == 0.0


def test_catalog_coverage_ignores_out_of_catalog_items() -> None:
    """Items recommended that are not in the catalogue do not count."""
    # only 'a' is in the catalog; 'z' is ignored -> 1 of 2 = 0.5.
    assert catalog_coverage([["a", "z"]], {"a", "b"}) == pytest.approx(0.5, abs=1e-12)
