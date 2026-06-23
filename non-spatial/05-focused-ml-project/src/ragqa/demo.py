"""One-command, dependency-light demo of the real RAG retrieval core.

This drives the *actual* pure-numpy retrieval core end to end on the bundled
portfolio corpus (``corpus/*.md``): it loads the documents, chunks them, builds
the TF-IDF :class:`~ragqa.index.Retriever`, evaluates retrieval against the
bundled QA set with recall@k and MRR, and answers a couple of sample questions
extractively (no API key, no LLM).

Everything is deterministic and depends only on numpy + stdlib, so it runs
anywhere including CI. The metrics are pinned by a test, so the README numbers
stay honest. Run it with ``python -m ragqa.cli demo`` or ``run_demo(seed=0)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragqa.chunk import chunk_documents
from ragqa.corpus import load_corpus, load_qa
from ragqa.evaluate import evaluate_retrieval
from ragqa.index import Retriever
from ragqa.pipeline import RAG

# The bundled corpus lives at the project root, two levels up from this file
# (src/ragqa/demo.py -> project/).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_DIR = _PROJECT_ROOT / "corpus"

# Chunking parameters. The corpus docs are short (1-2 paragraphs), so a modest
# window with overlap keeps each doc to a couple of retrievable chunks.
_CHUNK_SIZE = 60
_CHUNK_OVERLAP = 15

# Two sample questions answered extractively in the demo output.
_SAMPLE_QUESTIONS = (
    "Which project detects vegetation stress with Sentinel-2 NDVI anomalies?",
    "Where is feature drift detected with the Population Stability Index?",
)


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Build the retriever over the bundled corpus, evaluate it, and answer.

    Parameters
    ----------
    seed:
        Accepted for interface parity and reproducibility; the pipeline is fully
        deterministic, so the result does not depend on it.
    out_dir:
        Directory for ``eval.json`` and ``sample_answers.json`` (created if
        missing).

    Returns
    -------
    dict
        ``n_docs``, ``n_chunks``, ``vocab_size``, ``recall_at_3``, ``mrr``,
        ``sample_question`` and ``sample_top_doc``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Load + chunk the bundled portfolio docs --------------------------
    docs = load_corpus(_CORPUS_DIR)
    chunks = chunk_documents(docs, size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP)

    # --- Build the TF-IDF retriever (the star) ----------------------------
    retriever = Retriever().build(chunks)
    vocab_size = len(retriever.vectorizer.vocabulary_)

    # --- Evaluate retrieval against the bundled QA set --------------------
    qa_pairs = load_qa(_CORPUS_DIR / "qa_eval.json")
    metrics = evaluate_retrieval(retriever, qa_pairs, k=3)

    # --- Answer the sample questions extractively (free, no key) ----------
    rag = RAG(retriever, provider="extractive", use_mmr=True)
    sample_answers = []
    for q in _SAMPLE_QUESTIONS:
        out = rag.answer(q, k=3)
        top_doc = out["sources"][0] if out["sources"] else None
        sample_answers.append(
            {
                "question": q,
                "answer": out["answer"],
                "sources": out["sources"],
                "top_doc": top_doc,
            }
        )

    result: dict[str, Any] = {
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "vocab_size": vocab_size,
        "recall_at_3": round(metrics["recall_at_k"], 6),
        "mrr": round(metrics["mrr"], 6),
        "sample_question": sample_answers[0]["question"],
        "sample_top_doc": sample_answers[0]["top_doc"],
    }

    # --- Write artifacts ---------------------------------------------------
    with (out_path / "eval.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_docs": result["n_docs"],
                "n_chunks": result["n_chunks"],
                "vocab_size": result["vocab_size"],
                "recall_at_3": result["recall_at_3"],
                "mrr": result["mrr"],
                "n_questions": metrics["n"],
            },
            fh,
            indent=2,
        )
    with (out_path / "sample_answers.json").open("w", encoding="utf-8") as fh:
        json.dump(sample_answers, fh, indent=2)

    return result


if __name__ == "__main__":  # pragma: no cover - manual entry point
    print(json.dumps(run_demo(0), indent=2))
