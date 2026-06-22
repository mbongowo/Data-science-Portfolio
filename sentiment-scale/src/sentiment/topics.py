r"""TF-IDF for topic extraction (pure numpy).

A small, transparent TF-IDF used as the front end of topic extraction
(TF-IDF -> clustering -> top terms per cluster; the clustering itself lives in
the heavier modules). The formula is fixed and documented so the numbers are
reproducible and testable.

For a corpus of ``N`` documents and a vocabulary built from the unique tokens:

* **Term frequency** ``tf[d, t]`` is the raw count of term ``t`` in document
  ``d``.
* **Inverse document frequency** uses the smoothed form

  .. math::

      \mathrm{idf}(t) = \ln\!\frac{N + 1}{\mathrm{df}(t) + 1} + 1

  where ``df(t)`` is the number of documents containing ``t``. The ``+1`` terms
  (smoothing) keep the logarithm finite for a term that appears in every
  document and avoid a zero idf. This is the same smoothing scikit-learn uses
  with ``smooth_idf=True``.
* The TF-IDF weight is ``tf[d, t] * idf(t)``. No L2 row normalisation is applied
  here, so the returned values are exactly ``tf * idf`` and easy to verify by
  hand.

The vocabulary is the sorted set of tokens, so column order is deterministic.
"""

from __future__ import annotations

import numpy as np

from sentiment.clean import tokenize


def tfidf(docs: list[str]) -> tuple[np.ndarray, list[str]]:
    r"""Compute a TF-IDF matrix and vocabulary for ``docs``.

    Parameters
    ----------
    docs:
        A list of raw document strings. Each is tokenised with
        :func:`sentiment.clean.tokenize`.

    Returns
    -------
    matrix : numpy.ndarray
        An ``(N, V)`` float array of ``tf * idf`` weights, where ``N`` is the
        number of documents and ``V`` the vocabulary size. Row ``d`` column
        ``t`` is ``count(t in d) * idf(t)``.
    vocab : list[str]
        The sorted vocabulary; ``vocab[t]`` names column ``t``.

    Raises
    ------
    ValueError
        If ``docs`` is empty.

    Notes
    -----
    ``idf(t) = ln((N + 1) / (df(t) + 1)) + 1``. A term present in every document
    gets ``idf = ln((N+1)/(N+1)) + 1 = 1``; a rarer term gets a larger idf.

    Examples
    --------
    >>> matrix, vocab = tfidf(["good good", "bad"])
    >>> vocab
    ['bad', 'good']
    >>> import numpy as np
    >>> # "good" appears in 1 of 2 docs: idf = ln(3/2) + 1
    >>> float(matrix[0, 1].round(6)) == round(2 * (np.log(3 / 2) + 1), 6)
    True
    """
    if not docs:
        raise ValueError("tfidf requires at least one document.")

    tokenized = [tokenize(d) for d in docs]
    vocab = sorted({tok for toks in tokenized for tok in toks})
    index = {tok: j for j, tok in enumerate(vocab)}

    n = len(docs)
    v = len(vocab)
    tf = np.zeros((n, v), dtype=float)
    for i, toks in enumerate(tokenized):
        for tok in toks:
            tf[i, index[tok]] += 1.0

    df = (tf > 0).sum(axis=0)
    idf = np.log((n + 1) / (df + 1)) + 1.0
    return tf * idf, vocab
