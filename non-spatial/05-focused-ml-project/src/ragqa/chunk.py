"""Document chunking (pure stdlib).

Retrieval-augmented generation does not retrieve whole documents; it retrieves
*chunks*. A chunk is a short, self-contained window of text that is small enough
to be a precise retrieval unit but large enough to carry context. This module
splits a document into overlapping word windows.

Two parameters control the split:

* ``size`` — the number of words per chunk.
* ``overlap`` — how many words each chunk shares with the previous one. Overlap
  stops a sentence that straddles a boundary from being cut in half and lost to
  retrieval; the cost is a little redundancy.

The stride between chunk starts is ``size - overlap``. With ``size=5`` and
``overlap=2`` the starts are ``0, 3, 6, ...`` and consecutive chunks share two
words. The boundaries are deterministic and easy to derive by hand, so they are
pinned by a known-answer test.
"""

from __future__ import annotations


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split ``text`` into overlapping word windows.

    Parameters
    ----------
    text:
        The raw document text. Split on whitespace into words.
    size:
        Words per chunk (``size >= 1``).
    overlap:
        Words shared with the previous chunk (``0 <= overlap < size``).

    Returns
    -------
    list[str]
        The chunks, each a space-joined run of at most ``size`` words. An
        empty or whitespace-only ``text`` yields an empty list.

    Raises
    ------
    ValueError
        If ``size < 1`` or ``overlap`` is outside ``[0, size)``.

    Examples
    --------
    >>> chunk_text("a b c d e f g", size=3, overlap=1)
    ['a b c', 'c d e', 'e f g']
    """
    if size < 1:
        raise ValueError("size must be at least 1.")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size.")

    words = text.split()
    if not words:
        return []

    stride = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), stride):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            # The last full window already reached the end; stop so we do not
            # emit further windows that only repeat the tail.
            break
    return chunks


def chunk_documents(
    docs: list[dict],
    size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """Chunk a list of documents into retrieval units.

    Parameters
    ----------
    docs:
        Documents as dicts with at least ``doc_id`` and ``text`` keys (a
        ``title`` is carried through if present).
    size, overlap:
        Passed to :func:`chunk_text`.

    Returns
    -------
    list[dict]
        One dict per chunk with keys ``doc_id``, ``chunk_id`` (a running index
        within its document, starting at 0), ``text`` and, when available,
        ``title``. Documents that chunk to nothing contribute no entries.
    """
    out: list[dict] = []
    for doc in docs:
        pieces = chunk_text(doc["text"], size=size, overlap=overlap)
        for chunk_id, piece in enumerate(pieces):
            entry = {
                "doc_id": doc["doc_id"],
                "chunk_id": chunk_id,
                "text": piece,
            }
            if "title" in doc:
                entry["title"] = doc["title"]
            out.append(entry)
    return out
