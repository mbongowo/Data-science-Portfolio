"""The RAG pipeline: retrieve, then answer.

:class:`RAG` ties a built :class:`ragqa.index.Retriever` to an answer provider.
``answer(question, k)`` retrieves the top-``k`` chunks and then either quotes the
best one (the pure, free ``extractive`` default) or hands the chunks to an LLM
provider for a grounded generated answer.

Only the retrieve + extractive path runs in the tests; the LLM providers are
dispatched lazily inside :func:`ragqa.generate.llm_answer`, so importing or
testing this module never pulls in a provider SDK.
"""

from __future__ import annotations

from ragqa.generate import extractive_answer


class RAG:
    """Retrieval-augmented question answering over a corpus.

    Parameters
    ----------
    retriever:
        A built :class:`ragqa.index.Retriever`.
    provider:
        ``"extractive"`` (default, free, no key) quotes the best chunk;
        ``"openai"`` / ``"azure_openai"`` / ``"local"`` generate a grounded
        answer via :func:`ragqa.generate.llm_answer`.
    model:
        Optional model / deployment name passed to the LLM provider.
    use_mmr:
        Whether retrieval re-ranks candidates for diversity with MMR.
    """

    def __init__(
        self,
        retriever,
        provider: str = "extractive",
        model: str | None = None,
        use_mmr: bool = True,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.model = model
        self.use_mmr = use_mmr

    def answer(self, question: str, k: int = 4) -> dict:
        """Answer ``question`` from the corpus.

        Returns
        -------
        dict
            ``{"answer", "sources", "retrieved_chunks", "provider"}``. The
            retrieved chunks are the raw retrieval results (with scores) so the
            caller can show the evidence behind the answer.
        """
        retrieved = self.retriever.query(question, k=k, use_mmr=self.use_mmr)

        if self.provider == "extractive":
            result = extractive_answer(question, retrieved)
        else:
            # Lazy import keeps provider SDKs out of the import path and tests.
            from ragqa.generate import llm_answer

            result = llm_answer(
                question, retrieved, provider=self.provider, model=self.model
            )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": retrieved,
            "provider": result["provider"],
        }
