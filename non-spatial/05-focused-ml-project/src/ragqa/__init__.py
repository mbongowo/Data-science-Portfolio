"""ragqa: retrieval-augmented question answering over my portfolio docs.

A small, transparent RAG system. The runnable, CI-tested core is pure-Python /
numpy retrieval: chunk documents, build a TF-IDF index, retrieve the top-k
chunks by cosine similarity, optionally re-rank for diversity with Maximal
Marginal Relevance, and evaluate retrieval with recall@k and MRR. Answers
default to **extractive** — quote the best retrieved chunk with citations — which
needs no API key and cannot hallucinate.

Generative answers via an LLM (OpenAI, Azure OpenAI or a local transformers
model) are a pluggable, optional provider in :mod:`ragqa.generate`, imported
lazily and never touched by the test suite, so the core stays importable with
only numpy + stdlib.
"""

from __future__ import annotations

from ragqa.chunk import chunk_documents, chunk_text
from ragqa.corpus import load_corpus, load_qa
from ragqa.evaluate import evaluate_retrieval, mrr, recall_at_k
from ragqa.generate import extractive_answer
from ragqa.index import Retriever, TfidfVectorizer, cosine_similarity, mmr
from ragqa.pipeline import RAG

__all__ = [
    "chunk_text",
    "chunk_documents",
    "TfidfVectorizer",
    "cosine_similarity",
    "Retriever",
    "mmr",
    "recall_at_k",
    "mrr",
    "evaluate_retrieval",
    "extractive_answer",
    "RAG",
    "load_corpus",
    "load_qa",
    "__version__",
]

__version__ = "0.1.0"
