"""Command-line interface for the portfolio RAG system.

``demo`` runs the pure retrieval core (load -> chunk -> TF-IDF index -> evaluate
-> extractive answers) over the bundled corpus and is the CI-tested, runnable
contribution. ``index`` reports the index built over a corpus directory, and
``ask`` answers a question — extractively for free, or via an LLM provider when
``--provider`` is set and the matching key is in the environment. The LLM import
is lazy, so importing this CLI stays cheap and the tested core needs no provider
SDK.

Run ``python -m ragqa.cli demo`` to reproduce the result numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS = _PROJECT_ROOT / "corpus"


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the bundled retrieval demo and print the summary."""
    from ragqa.demo import run_demo

    result = run_demo(seed=args.seed, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


def _build_retriever(corpus_dir: str):
    """Load + chunk a corpus directory and build a Retriever (helper)."""
    from ragqa.chunk import chunk_documents
    from ragqa.corpus import load_corpus
    from ragqa.index import Retriever

    docs = load_corpus(corpus_dir)
    chunks = chunk_documents(docs, size=60, overlap=15)
    return Retriever().build(chunks), docs, chunks


def _cmd_index(args: argparse.Namespace) -> int:
    """Build the index over a corpus and report its shape."""
    retriever, docs, chunks = _build_retriever(args.corpus)
    print(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "n_docs": len(docs),
                "n_chunks": len(chunks),
                "vocab_size": len(retriever.vectorizer.vocabulary_),
            },
            indent=2,
        )
    )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question over the corpus (extractive by default; LLM optional)."""
    from ragqa.pipeline import RAG

    retriever, _, _ = _build_retriever(args.corpus)
    rag = RAG(retriever, provider=args.provider, model=args.model)
    out = rag.answer(args.question, k=args.k)
    print(out["answer"])
    print(f"\nSources: {', '.join(map(str, out['sources']))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI."""
    parser = argparse.ArgumentParser(
        prog="ragqa",
        description="Ask questions over a portfolio of project docs (RAG).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the bundled retrieval demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out-dir", default="outputs")
    demo.set_defaults(func=_cmd_demo)

    index = sub.add_parser("index", help="build the index over a corpus and report it")
    index.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    index.set_defaults(func=_cmd_index)

    ask = sub.add_parser("ask", help="answer a question over the corpus")
    ask.add_argument("question", help="the natural-language question")
    ask.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    ask.add_argument("--k", type=int, default=4, help="chunks to retrieve")
    ask.add_argument(
        "--provider",
        default="extractive",
        choices=["extractive", "openai", "azure_openai", "local"],
        help="answer provider (LLM providers need a key/model in the env)",
    )
    ask.add_argument("--model", default=None, help="LLM model / deployment name")
    ask.set_defaults(func=_cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``ragqa``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
