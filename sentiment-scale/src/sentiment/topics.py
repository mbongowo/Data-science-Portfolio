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


def nmf(
    X: np.ndarray,
    k: int,
    iters: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Non-negative matrix factorisation by multiplicative updates (pure numpy).

    Factorise a non-negative ``(N, V)`` matrix ``X`` (e.g. a TF-IDF or
    bag-of-words matrix) into two non-negative factors ``W`` ``(N, k)`` and ``H``
    ``(k, V)`` such that ``W @ H`` approximately reconstructs ``X``. The ``k``
    rows of ``H`` are interpretable as *topics* over the vocabulary, and row
    ``d`` of ``W`` is document ``d``'s weight on each topic — a transparent topic
    model that, unlike SVD, never produces negative loadings.

    The optimisation uses Lee & Seung's multiplicative update rules for the
    Frobenius reconstruction error :math:`\lVert X - WH \rVert_F^2`:

    .. math::

        H \leftarrow H \odot \frac{W^\top X}{W^\top W H},\qquad
        W \leftarrow W \odot \frac{X H^\top}{W H H^\top}

    Both updates preserve non-negativity (a non-negative matrix times a
    non-negative ratio stays non-negative) and do not increase the error, so the
    reconstruction error is monotone non-increasing.

    Parameters
    ----------
    X:
        A non-negative ``(N, V)`` matrix.
    k:
        Number of latent topics / factors (``1 <= k``).
    iters:
        Number of multiplicative-update iterations.
    seed:
        Seed for ``numpy.random.default_rng``; the random initial factors (and
        thus the result) are reproducible.

    Returns
    -------
    (W, H) : tuple[numpy.ndarray, numpy.ndarray]
        Non-negative factors of shapes ``(N, k)`` and ``(k, V)``.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, contains negative entries, or ``k < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    >>> W, H = nmf(X, k=2, iters=300, seed=0)
    >>> bool((W >= 0).all() and (H >= 0).all())
    True
    >>> float(np.linalg.norm(X - W @ H)) < 0.1
    True
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if (X < 0).any():
        raise ValueError("nmf requires a non-negative matrix X.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    n, v = X.shape
    rng = np.random.default_rng(seed)
    # Scale the initial factors so W @ H starts near the magnitude of X; this
    # makes the multiplicative updates converge in fewer iterations.
    scale = np.sqrt(X.mean() / k) if X.mean() > 0 else 1.0
    w = rng.random((n, k)) * scale
    h = rng.random((k, v)) * scale

    eps = 1e-10
    for _ in range(iters):
        # Update H, then W (using the freshly updated H), per Lee & Seung.
        h *= (w.T @ X) / (w.T @ w @ h + eps)
        w *= (X @ h.T) / (w @ (h @ h.T) + eps)

    return w, h
