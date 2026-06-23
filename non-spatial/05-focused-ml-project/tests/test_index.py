"""Known-answer tests for the TF-IDF retrieval core."""

from __future__ import annotations

import numpy as np

from ragqa.index import Retriever, TfidfVectorizer, cosine_similarity, mmr


def test_tfidf_idf_values_hand_derived():
    # N = 3 docs. df: bird1, cat1, dog1, flew1, sat2, the3.
    # idf = ln((N+1)/(df+1)) + 1.
    docs = ["the cat sat", "the dog sat", "the bird flew"]
    vec = TfidfVectorizer().fit(docs)
    assert vec.vocabulary_ == {
        "bird": 0,
        "cat": 1,
        "dog": 2,
        "flew": 3,
        "sat": 4,
        "the": 5,
    }
    expected_idf = np.array(
        [
            np.log(4 / 2) + 1,  # bird (df 1)
            np.log(4 / 2) + 1,  # cat  (df 1)
            np.log(4 / 2) + 1,  # dog  (df 1)
            np.log(4 / 2) + 1,  # flew (df 1)
            np.log(4 / 3) + 1,  # sat  (df 2)
            np.log(4 / 4) + 1,  # the  (df 3) -> 1.0
        ]
    )
    assert np.allclose(vec.idf_, expected_idf)


def test_tfidf_rows_are_l2_normalised():
    docs = ["the cat sat", "the dog sat", "the bird flew"]
    M = TfidfVectorizer().fit_transform(docs)
    norms = np.linalg.norm(M, axis=1)
    assert np.allclose(norms, 1.0)


def test_cosine_similarity_known_values():
    docs = ["the cat sat", "the dog sat", "the bird flew"]
    M = TfidfVectorizer().fit_transform(docs)
    sims = cosine_similarity(M[0], M)
    # Doc 0 vs itself is 1; vs doc 1 (shares "the", "sat") > vs doc 2 (only "the").
    assert np.isclose(sims[0], 1.0)
    assert sims[1] > sims[2] > 0.0
    assert np.isclose(sims[1], 0.48112, atol=1e-4)


def test_cosine_similarity_zero_vector_is_zero():
    B = np.array([[1.0, 0.0], [0.0, 0.0]])
    sims = cosine_similarity(np.array([1.0, 0.0]), B)
    assert sims[0] == 1.0
    assert sims[1] == 0.0  # zero document vector -> 0, no NaN


def test_retriever_returns_obviously_relevant_chunk_top1():
    chunks = [
        {"doc_id": "weather", "chunk_id": 0, "text": "rainfall drought and weather"},
        {"doc_id": "graph", "chunk_id": 0, "text": "pagerank graph nodes and edges"},
        {"doc_id": "movies", "chunk_id": 0, "text": "movie recommender ratings users"},
    ]
    r = Retriever().build(chunks)
    top = r.query("pagerank over a graph", k=1)
    assert top[0]["doc_id"] == "graph"
    assert top[0]["score"] > 0


def test_retriever_empty_or_oov_query_returns_empty():
    chunks = [{"doc_id": "d", "chunk_id": 0, "text": "alpha beta gamma"}]
    r = Retriever().build(chunks)
    assert r.query("", k=3) == []
    assert r.query("zzz qqq", k=3) == []  # no token in vocabulary


def test_mmr_deduplicates_near_identical_chunks():
    # Three chunks: two near-identical "apple" docs and one distinct "banana".
    docs = [
        "apple apple apple fruit red",
        "apple apple apple fruit red sweet",
        "banana fruit yellow",
    ]
    vec = TfidfVectorizer()
    M = vec.fit_transform(docs)
    query = vec.transform(["apple fruit"])[0]
    # With diversity weighting, the second pick should be the distinct banana
    # doc (index 2), not the near-duplicate of the first apple doc.
    picks = mmr(query, M, k=2, lambda_=0.5)
    assert picks[0] in (0, 1)
    assert picks[1] == 2


def test_retriever_query_use_mmr_path_runs():
    chunks = [
        {"doc_id": "a", "chunk_id": 0, "text": "apple apple fruit red"},
        {"doc_id": "a", "chunk_id": 1, "text": "apple apple fruit red sweet"},
        {"doc_id": "b", "chunk_id": 0, "text": "banana fruit yellow"},
    ]
    r = Retriever().build(chunks)
    results = r.query("apple fruit", k=2, use_mmr=True)
    assert len(results) == 2
    assert {res["doc_id"] for res in results} == {"a", "b"}
