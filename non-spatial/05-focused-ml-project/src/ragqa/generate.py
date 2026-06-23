"""Answer generation: a free extractive default and an optional LLM provider.

There are two ways to turn retrieved chunks into an answer:

* :func:`extractive_answer` — **pure, free, testable**. It quotes the single
  most relevant retrieved chunk verbatim and lists the source documents. No API
  key, no model download; it never invents text, so it cannot hallucinate beyond
  the corpus. This is the default and the only path the tests exercise.
* :func:`llm_answer` — **optional, lazy**. It dispatches to OpenAI, Azure OpenAI
  or a local transformers pipeline, building a grounded prompt that pins the
  model to the retrieved contexts and asks it to cite sources. It needs an API
  key (or a downloaded model) and the relevant client library, both imported
  *inside* the function so this module stays import-light and the test suite
  never touches a provider SDK.
"""

from __future__ import annotations


def extractive_answer(query: str, contexts: list[dict]) -> dict:
    """Answer ``query`` by quoting the most relevant retrieved chunk.

    Parameters
    ----------
    query:
        The user's question (used only to echo it back; the ranking has already
        been done by the retriever).
    contexts:
        Retrieved chunk dicts, best-first, each with at least ``doc_id`` and
        ``text`` (a ``title`` and ``score`` are used when present).

    Returns
    -------
    dict
        ``{"answer": str, "sources": list, "provider": "extractive"}``. With no
        contexts the answer states that nothing relevant was found and
        ``sources`` is empty.
    """
    if not contexts:
        return {
            "answer": "No relevant passage was found in the corpus for that question.",
            "sources": [],
            "provider": "extractive",
        }

    top = contexts[0]
    # Sources: the distinct documents the contexts came from, best-first.
    sources: list = []
    for c in contexts:
        if c["doc_id"] not in sources:
            sources.append(c["doc_id"])

    title = top.get("title")
    label = f"{title} ({top['doc_id']})" if title else str(top["doc_id"])
    answer = (
        f"{top['text'].strip()}\n\n"
        f"(Quoted from: {label}. Sources: {', '.join(map(str, sources))}.)"
    )
    return {"answer": answer, "sources": sources, "provider": "extractive"}


def _grounded_prompt(query: str, contexts: list[dict]) -> str:
    """Build a prompt that pins an LLM to the retrieved contexts."""
    blocks = []
    for c in contexts:
        title = c.get("title", c["doc_id"])
        blocks.append(f"[source: {c['doc_id']} | {title}]\n{c['text']}")
    context_text = "\n\n".join(blocks)
    return (
        "You are a question-answering assistant for a data-science portfolio. "
        "Answer the question using ONLY the context passages below. If the "
        "answer is not contained in them, say you do not know. Cite the source "
        "ids you used in parentheses.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )


def llm_answer(
    query: str,
    contexts: list[dict],
    provider: str = "openai",
    model: str | None = None,
) -> dict:
    """Generate a grounded answer with an optional LLM provider (lazy).

    This is **opt-in** and not used by the tests. Each provider's client library
    and credentials are required and imported inside the function:

    * ``provider="openai"`` — needs ``openai`` and ``OPENAI_API_KEY``.
    * ``provider="azure_openai"`` — needs ``openai`` (``AzureOpenAI`` client) and
      ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_KEY`` /
      ``AZURE_OPENAI_DEPLOYMENT``.
    * ``provider="local"`` — needs ``transformers`` and a local text-generation
      model (no network, no key).

    Parameters
    ----------
    query, contexts:
        The question and the retrieved chunks to ground on.
    provider:
        ``"openai"``, ``"azure_openai"`` or ``"local"``.
    model:
        Model / deployment name; sensible defaults per provider.

    Returns
    -------
    dict
        ``{"answer": str, "sources": list, "provider": provider}``.
    """
    if not contexts:
        return {
            "answer": "No relevant passage was found in the corpus for that question.",
            "sources": [],
            "provider": provider,
        }

    prompt = _grounded_prompt(query, contexts)
    sources: list = []
    for c in contexts:
        if c["doc_id"] not in sources:
            sources.append(c["doc_id"])

    if provider == "openai":
        import os

        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content

    elif provider == "azure_openai":
        import os

        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-06-01",
        )
        resp = client.chat.completions.create(
            model=model or os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content

    elif provider == "local":
        from transformers import pipeline

        gen = pipeline("text-generation", model=model or "gpt2")
        text = gen(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        # Return only the continuation after the prompt.
        text = text[len(prompt) :].strip()

    else:
        raise ValueError(
            f"Unknown provider {provider!r}; "
            "expected 'openai', 'azure_openai' or 'local'."
        )

    return {"answer": text, "sources": sources, "provider": provider}
