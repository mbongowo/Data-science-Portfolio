"""Known-answer tests for chunking: exact boundaries and overlap."""

from __future__ import annotations

import pytest

from ragqa.chunk import chunk_documents, chunk_text


def test_chunk_text_exact_boundaries_with_overlap():
    # 7 words, size 3, overlap 1 -> stride 2 -> starts 0, 2, 4 (start 4 reaches
    # the end at index 7, so no redundant trailing fragment is emitted).
    text = "a b c d e f g"
    assert chunk_text(text, size=3, overlap=1) == ["a b c", "c d e", "e f g"]


def test_chunk_text_no_overlap_tiles_cleanly():
    text = "one two three four five six"
    assert chunk_text(text, size=2, overlap=0) == [
        "one two",
        "three four",
        "five six",
    ]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", size=5, overlap=2) == []
    assert chunk_text("   ", size=5, overlap=2) == []


def test_chunk_text_text_shorter_than_size():
    assert chunk_text("just two", size=10, overlap=3) == ["just two"]


def test_chunk_text_validates_params():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", size=3, overlap=3)  # overlap must be < size


def test_chunk_documents_ids_are_running_per_doc():
    docs = [
        {"doc_id": "d1", "title": "D1", "text": "a b c d e"},
        {"doc_id": "d2", "title": "D2", "text": "x y"},
    ]
    chunks = chunk_documents(docs, size=2, overlap=0)
    # d1: "a b", "c d", "e"  -> chunk_ids 0,1,2 ; d2: "x y" -> chunk_id 0
    assert [(c["doc_id"], c["chunk_id"]) for c in chunks] == [
        ("d1", 0),
        ("d1", 1),
        ("d1", 2),
        ("d2", 0),
    ]
    assert chunks[0]["text"] == "a b"
    assert chunks[0]["title"] == "D1"
