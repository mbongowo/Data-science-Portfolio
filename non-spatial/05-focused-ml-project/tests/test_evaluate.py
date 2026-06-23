"""Known-answer tests for retrieval evaluation metrics."""

from __future__ import annotations

from ragqa.evaluate import evaluate_retrieval, mrr, recall_at_k


def test_recall_at_k_true_and_false():
    assert recall_at_k(["a", "b", "c"], "b", k=2) == 1.0
    assert recall_at_k(["a", "b", "c"], "c", k=2) == 0.0  # c is at index 2, k=2
    assert recall_at_k(["a", "b", "c"], "c", k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], "z", k=3) == 0.0


def test_mrr_is_reciprocal_of_rank():
    assert mrr(["a", "b", "c"], "a") == 1.0
    assert mrr(["a", "b", "c"], "b") == 0.5
    assert mrr(["a", "b", "c"], "c") == 1.0 / 3.0
    assert mrr(["a", "b", "c"], "z") == 0.0


class _FakeRetriever:
    """Returns a fixed ranking of doc_ids per question for deterministic eval."""

    def __init__(self, rankings):
        self._rankings = rankings

    def query(self, question, k=3, **kwargs):
        return [
            {"doc_id": d, "chunk_id": 0, "text": "", "score": 0.0}
            for d in self._rankings[question][:k]
        ]


def test_evaluate_retrieval_aggregates_correctly():
    # q1: relevant "x" at rank 1 -> recall 1, rr 1.0
    # q2: relevant "y" at rank 2 -> recall 1, rr 0.5
    # q3: relevant "z" absent from top-3 -> recall 0, rr 0.0
    rankings = {
        "q1": ["x", "a", "b"],
        "q2": ["a", "y", "b"],
        "q3": ["a", "b", "c"],
    }
    retr = _FakeRetriever(rankings)
    pairs = [("q1", "x"), ("q2", "y"), ("q3", "z")]
    out = evaluate_retrieval(retr, pairs, k=3)
    assert out["n"] == 3
    assert out["k"] == 3
    assert out["recall_at_k"] == 2 / 3
    assert abs(out["mrr"] - (1.0 + 0.5 + 0.0) / 3) < 1e-12


def test_evaluate_retrieval_empty():
    out = evaluate_retrieval(_FakeRetriever({}), [], k=3)
    assert out == {"recall_at_k": 0.0, "mrr": 0.0, "k": 3, "n": 0}
