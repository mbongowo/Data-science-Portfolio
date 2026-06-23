"""Retrieval evaluation metrics (pure stdlib/numpy).

Retrieval is the honest, measurable part of a RAG system: given a question with
a known relevant document, does the retriever surface that document, and how
high does it rank it? Two standard metrics answer that.

* **Recall@k** — did the relevant document appear anywhere in the top ``k``?
  Averaged over a set of questions it is the fraction for which retrieval put the
  right document within reach of the answer step.
* **MRR (Mean Reciprocal Rank)** — ``1 / rank`` of the first relevant document
  (rank 1 -> 1.0, rank 2 -> 0.5, ...), or 0 if it is absent. It rewards ranking
  the right document *higher*, not merely including it.

Both are defined against a single ``relevant_doc_id`` per question, which matches
the bundled QA set where each question is answered by one portfolio doc.
"""

from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list, relevant_doc_id, k: int) -> float:
    """1.0 if ``relevant_doc_id`` is among the first ``k`` retrieved, else 0.0.

    Examples
    --------
    >>> recall_at_k(["a", "b", "c"], "b", k=2)
    1.0
    >>> recall_at_k(["a", "b", "c"], "c", k=2)
    0.0
    """
    if k < 0:
        raise ValueError("k must be non-negative.")
    return 1.0 if relevant_doc_id in retrieved_doc_ids[:k] else 0.0


def mrr(retrieved_doc_ids: list, relevant_doc_id) -> float:
    """Reciprocal rank of ``relevant_doc_id`` in ``retrieved_doc_ids``.

    Returns ``1 / rank`` for the first (1-based) position of the relevant id, or
    ``0.0`` if it never appears.

    Examples
    --------
    >>> mrr(["a", "b", "c"], "a")
    1.0
    >>> mrr(["a", "b", "c"], "b")
    0.5
    >>> mrr(["a", "b", "c"], "z")
    0.0
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(retriever, qa_pairs: list[tuple], k: int = 3) -> dict:
    """Aggregate recall@k and MRR over ``qa_pairs``.

    Parameters
    ----------
    retriever:
        A built :class:`ragqa.index.Retriever` (or anything with a ``query``
        method returning chunk dicts that carry ``doc_id``).
    qa_pairs:
        A list of ``(question, relevant_doc_id)`` tuples.
    k:
        Cut-off for recall and the number of chunks to retrieve per question.

    Returns
    -------
    dict
        ``{"recall_at_k": float, "mrr": float, "k": k, "n": len(qa_pairs)}``.
        Empty input yields zero metrics. For MRR the retrieved doc-id list is
        de-duplicated to first occurrence, so the reciprocal rank reflects the
        first *document* rather than the first *chunk*.
    """
    if not qa_pairs:
        return {"recall_at_k": 0.0, "mrr": 0.0, "k": k, "n": 0}

    recalls: list[float] = []
    rrs: list[float] = []
    for question, relevant_doc_id in qa_pairs:
        results = retriever.query(question, k=k)
        doc_ids = [r["doc_id"] for r in results]
        # Collapse to first occurrence per document for a document-level rank.
        seen: list = []
        for d in doc_ids:
            if d not in seen:
                seen.append(d)
        recalls.append(recall_at_k(seen, relevant_doc_id, k))
        rrs.append(mrr(seen, relevant_doc_id))

    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "k": k,
        "n": len(qa_pairs),
    }
