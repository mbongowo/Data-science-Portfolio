"""Streamlit entry point for the Portfolio RAG chat app.

Run locally with::

    streamlit run app/streamlit_app.py

The app builds the pure-numpy TF-IDF retriever once (cached) over the bundled
portfolio corpus, takes a natural-language question, and returns an answer with
cited source documents and the retrieved chunks and their scores. The default
provider is **extractive** — it quotes the best retrieved chunk and needs no API
key. A sidebar lets you switch to OpenAI or Azure OpenAI for a generated grounded
answer when the matching key is present in the environment.
"""

from __future__ import annotations

# When a host runs this file directly (Streamlit Community Cloud runs
# app/streamlit_app.py), only this file's directory is on sys.path, so the
# `ragqa` package under ../src is not importable. Put it on the path.
import sys as _sys
from pathlib import Path as _Path

_app_dir = _Path(__file__).resolve().parent
_project_root = _app_dir.parent
_src_dir = _project_root / "src"
for _p in (str(_src_dir), str(_project_root)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import os

import streamlit as st

from ragqa.chunk import chunk_documents
from ragqa.corpus import load_corpus
from ragqa.index import Retriever
from ragqa.pipeline import RAG

_CORPUS_DIR = _project_root / "corpus"

st.set_page_config(
    page_title="Portfolio RAG",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _build_retriever():
    """Load + chunk the bundled corpus and build the TF-IDF retriever (cached)."""
    docs = load_corpus(_CORPUS_DIR)
    chunks = chunk_documents(docs, size=60, overlap=15)
    retriever = Retriever().build(chunks)
    return retriever, len(docs), len(chunks)


def _provider_available(provider: str) -> bool:
    """True if the env keys required by ``provider`` are present."""
    if provider == "OpenAI":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "Azure OpenAI":
        return all(
            os.environ.get(k)
            for k in (
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_DEPLOYMENT",
            )
        )
    return True  # extractive needs nothing


_PROVIDER_IDS = {
    "Extractive (no key)": "extractive",
    "OpenAI": "openai",
    "Azure OpenAI": "azure_openai",
}


def main() -> None:
    st.title("📚 Portfolio RAG")
    st.write(
        "Ask a natural-language question across my data-science portfolio. The "
        "app retrieves the most relevant project docs with a pure-numpy TF-IDF "
        "index and answers with citations."
    )
    st.caption(
        "The default **Extractive** provider quotes the best retrieved passage "
        "and needs no API key. OpenAI / Azure OpenAI generate a grounded answer "
        "when their keys are set in the environment."
    )

    retriever, n_docs, n_chunks = _build_retriever()

    with st.sidebar:
        st.header("Settings")
        provider_label = st.radio(
            "Answer provider",
            list(_PROVIDER_IDS.keys()),
            index=0,
            help="Extractive is free and needs no key.",
        )
        k = st.slider("Chunks to retrieve (k)", 1, 8, 4)
        use_mmr = st.checkbox("Diversify with MMR", value=True)
        st.divider()
        st.caption(f"Corpus: {n_docs} docs, {n_chunks} chunks.")

        if provider_label != "Extractive (no key)" and not _provider_available(
            provider_label
        ):
            st.warning(
                f"{provider_label} keys not found in the environment — falling "
                "back to the free Extractive provider."
            )

    # Resolve the effective provider, falling back to extractive if no key.
    if provider_label != "Extractive (no key)" and _provider_available(provider_label):
        provider = _PROVIDER_IDS[provider_label]
    else:
        provider = "extractive"

    question = st.text_input(
        "Your question",
        placeholder="e.g. Which project detects floods from SAR imagery?",
    )

    if st.button("Ask", type="primary") and question.strip():
        rag = RAG(retriever, provider=provider, use_mmr=use_mmr)
        out = rag.answer(question, k=k)

        st.subheader("Answer")
        st.write(out["answer"])

        if out["sources"]:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in out["sources"]))

        st.subheader("Retrieved chunks")
        for c in out["retrieved_chunks"]:
            title = c.get("title", c["doc_id"])
            with st.expander(f"{title}  ({c['doc_id']}) — score {c['score']:.3f}"):
                st.write(c["text"])

    st.divider()
    st.caption(
        "Retrieval is lexical TF-IDF cosine similarity with optional MMR "
        "diversity. Swap in dense embeddings to improve semantic recall."
    )


if __name__ == "__main__":
    main()
else:
    main()
