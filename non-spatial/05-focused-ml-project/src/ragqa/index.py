r"""Pure-numpy TF-IDF retrieval core — the heart of the RAG pipeline.

Given a set of chunks, this module builds a TF-IDF vector for each chunk and
retrieves the chunks closest to a query by cosine similarity. It is deliberately
lexical and transparent: every number can be derived by hand, so the behaviour
is pinned by known-answer tests and runs anywhere with only numpy.

TF-IDF weighting
----------------
For a corpus of ``N`` documents (here, chunks) and a vocabulary of the unique
tokens, the weight of term ``t`` in document ``d`` is

.. math::

    w(d, t) = \mathrm{tf}(d, t)\;\big(\ln\tfrac{N + 1}{\mathrm{df}(t) + 1} + 1\big)

where ``tf(d, t)`` is the raw count of ``t`` in ``d`` and ``df(t)`` is the number
of documents containing ``t``. The ``+1`` smoothing keeps the logarithm finite
for a term that appears in every document and avoids a zero idf — the same
``smooth_idf=True`` convention scikit-learn uses. Each document row is then
**L2-normalised** so that a dot product between two rows is their cosine
similarity, which keeps long and short chunks comparable.

Maximal Marginal Relevance (MMR)
--------------------------------
Pure top-k by similarity can return several near-duplicate chunks that all match
the query but say the same thing. MMR re-ranks for *diversity*: at each step it
picks the candidate that maximises

.. math::

    \lambda\,\mathrm{sim}(q, c) - (1 - \lambda)\max_{s \in S}\mathrm{sim}(c, s)

balancing relevance to the query ``q`` against redundancy with the already
selected set ``S``. ``lambda_=1`` is plain relevance; ``lambda_=0`` is pure
diversity; ``0.5`` trades them off evenly.
"""

from __future__ import annotations

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on runs of letters/digits (drops punctuation)."""
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity of a query vector ``a`` against every row of ``B``.

    Parameters
    ----------
    a:
        A 1-D query vector of length ``V``.
    B:
        A 2-D ``(M, V)`` matrix of document vectors.

    Returns
    -------
    numpy.ndarray
        A length-``M`` array; entry ``i`` is ``cos(a, B[i])``. Zero vectors
        yield a similarity of 0 (no division by zero).
    """
    a = np.asarray(a, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be a 2-D matrix of document vectors.")
    a_norm = np.linalg.norm(a)
    b_norms = np.linalg.norm(B, axis=1)
    denom = b_norms * a_norm
    dots = B @ a
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 0, dots / denom, 0.0)
    return sims


class TfidfVectorizer:
    """Fit/transform TF-IDF vectoriser (pure numpy).

    The vocabulary is the sorted set of tokens seen at ``fit`` time, so column
    order is deterministic. ``transform`` ignores out-of-vocabulary tokens.
    Rows are L2-normalised, so a dot product of two transformed rows is their
    cosine similarity.
    """

    def __init__(self) -> None:
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, docs: list[str]) -> TfidfVectorizer:
        """Learn the vocabulary and idf weights from ``docs``."""
        if not docs:
            raise ValueError("Cannot fit on an empty corpus.")
        tokenized = [_tokenize(d) for d in docs]
        vocab = sorted({tok for toks in tokenized for tok in toks})
        if not vocab:
            raise ValueError("Corpus has no tokens after tokenisation.")
        self.vocabulary_ = {tok: j for j, tok in enumerate(vocab)}

        n = len(docs)
        v = len(vocab)
        df = np.zeros(v, dtype=float)
        for toks in tokenized:
            for tok in set(toks):
                df[self.vocabulary_[tok]] += 1.0
        self.idf_ = np.log((n + 1) / (df + 1)) + 1.0
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        """Map ``docs`` to L2-normalised TF-IDF rows ``(len(docs), V)``."""
        if self.idf_ is None:
            raise ValueError("Vectorizer must be fitted before transform.")
        v = len(self.vocabulary_)
        rows = np.zeros((len(docs), v), dtype=float)
        for i, doc in enumerate(docs):
            for tok in _tokenize(doc):
                j = self.vocabulary_.get(tok)
                if j is not None:
                    rows[i, j] += 1.0
        rows *= self.idf_  # tf * idf, broadcast over columns
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            rows = np.where(norms > 0, rows / norms, 0.0)
        return rows

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        """Convenience: :meth:`fit` then :meth:`transform` on the same docs."""
        return self.fit(docs).transform(docs)


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> list[int]:
    """Maximal Marginal Relevance selection of ``k`` row indices.

    Greedily selects indices into ``doc_vecs`` that are relevant to
    ``query_vec`` while penalising redundancy with the already-selected rows.

    Parameters
    ----------
    query_vec:
        A 1-D query vector.
    doc_vecs:
        A 2-D ``(M, V)`` matrix of candidate document vectors.
    k:
        How many indices to return (clamped to ``M``).
    lambda_:
        Relevance/diversity trade-off in ``[0, 1]``. ``1`` is pure relevance,
        ``0`` is pure diversity.

    Returns
    -------
    list[int]
        The selected row indices, in selection order.
    """
    doc_vecs = np.asarray(doc_vecs, dtype=float)
    if doc_vecs.ndim != 2:
        raise ValueError("doc_vecs must be a 2-D matrix.")
    m = doc_vecs.shape[0]
    k = min(k, m)
    if k <= 0:
        return []

    query_sim = cosine_similarity(query_vec, doc_vecs)
    selected: list[int] = []
    candidates = list(range(m))

    # Pairwise similarity between candidates, computed lazily as needed via dot
    # products on (already L2-normalised) rows.
    while candidates and len(selected) < k:
        if not selected:
            best = max(candidates, key=lambda i: query_sim[i])
        else:
            sel_mat = doc_vecs[selected]
            best, best_score = None, -np.inf
            for i in candidates:
                redundancy = float(np.max(sel_mat @ doc_vecs[i]))
                score = lambda_ * query_sim[i] - (1 - lambda_) * redundancy
                if score > best_score:
                    best, best_score = i, score
        selected.append(best)
        candidates.remove(best)
    return selected


class Retriever:
    """TF-IDF cosine retriever over a list of chunks.

    Build it once on the chunks, then ``query`` it with natural-language text to
    get the top-k most similar chunks with their cosine scores and source
    ``doc_id``s. Set ``use_mmr=True`` to re-rank the top candidates for
    diversity.
    """

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.vectorizer = TfidfVectorizer()
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[dict]) -> Retriever:
        """Fit TF-IDF over the ``text`` field of ``chunks``."""
        if not chunks:
            raise ValueError("Cannot build a Retriever on zero chunks.")
        self.chunks = list(chunks)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        return self

    def query(
        self,
        text: str,
        k: int = 5,
        use_mmr: bool = False,
        lambda_: float = 0.5,
        mmr_pool: int = 20,
    ) -> list[dict]:
        """Retrieve the top-``k`` chunks for ``text``.

        Parameters
        ----------
        text:
            The query string.
        k:
            Number of chunks to return.
        use_mmr:
            If true, take the top ``mmr_pool`` by cosine similarity and re-rank
            them with :func:`mmr` for diversity before truncating to ``k``.
        lambda_:
            MMR relevance/diversity trade-off (only when ``use_mmr``).
        mmr_pool:
            Candidate pool size fed to MMR.

        Returns
        -------
        list[dict]
            Each result is the original chunk dict augmented with a ``score``
            (cosine similarity to the query). Ordered best-first. An empty or
            all-out-of-vocabulary query returns ``[]``.
        """
        if self.matrix is None:
            raise ValueError("Retriever must be built before querying.")
        if not text or not text.strip():
            return []

        q = self.vectorizer.transform([text])[0]
        if np.linalg.norm(q) == 0:
            return []  # no query token is in the vocabulary

        sims = cosine_similarity(q, self.matrix)

        if use_mmr:
            pool = np.argsort(sims)[::-1][:mmr_pool]
            order_in_pool = mmr(q, self.matrix[pool], k=k, lambda_=lambda_)
            chosen = [int(pool[i]) for i in order_in_pool]
        else:
            chosen = [int(i) for i in np.argsort(sims)[::-1][:k]]

        results: list[dict] = []
        for i in chosen:
            entry = dict(self.chunks[i])
            entry["score"] = float(sims[i])
            results.append(entry)
        return results
