"""Tests for the pure extractive answer path (no LLM, no network)."""

from __future__ import annotations

from ragqa.generate import extractive_answer


def test_extractive_answer_quotes_top_chunk_and_lists_sources():
    contexts = [
        {"doc_id": "eo-monitor", "title": "EO Monitor", "text": "NDVI anomaly map.",
         "score": 0.9},
        {"doc_id": "change-detection", "title": "Change Detection",
         "text": "Otsu flood mapping.", "score": 0.4},
    ]
    out = extractive_answer("vegetation stress?", contexts)
    assert out["provider"] == "extractive"
    # The answer quotes the most relevant (first) chunk's text verbatim.
    assert "NDVI anomaly map." in out["answer"]
    # Sources are the distinct doc_ids, best-first.
    assert out["sources"] == ["eo-monitor", "change-detection"]
    assert "EO Monitor" in out["answer"]  # title is shown in the citation


def test_extractive_answer_dedupes_sources_from_same_doc():
    contexts = [
        {"doc_id": "graph", "title": "Graph", "text": "pagerank.", "score": 0.8},
        {"doc_id": "graph", "title": "Graph", "text": "modularity.", "score": 0.7},
        {"doc_id": "als", "title": "ALS", "text": "recommender.", "score": 0.3},
    ]
    out = extractive_answer("graph?", contexts)
    assert out["sources"] == ["graph", "als"]


def test_extractive_answer_empty_context_guard():
    out = extractive_answer("anything?", [])
    assert out["sources"] == []
    assert "No relevant passage" in out["answer"]
    assert out["provider"] == "extractive"
