r"""A trained logistic-regression classifier as a model alternative to the lexicon.

The lexicon scorer (:func:`sentiment.lexicon.score_text`) is fixed: it knows only
the valences it is handed. This module is the *learned* counterpart — a binary
logistic-regression classifier implemented in pure numpy that is **fit** from
labelled examples rather than read from a table. It turns documents into
bag-of-words count vectors over a learned vocabulary and minimises the
mean binary cross-entropy by batch gradient descent.

The model is intentionally small and transparent so its behaviour is checkable:

* :func:`bag_of_words` builds an ``(N, V)`` integer count matrix and a sorted
  vocabulary from a list of documents (same tokeniser as the rest of the core).
* :class:`LogisticRegression` implements ``fit`` / ``predict_proba`` / ``predict``
  with the standard sigmoid model

  .. math::

      p(y=1 \mid x) = \sigma(x \cdot w + b),\qquad \sigma(z) = \frac{1}{1+e^{-z}}

  fit by gradient descent on the cross-entropy loss with optional L2
  regularisation. On a linearly separable training set it reaches 100% train
  accuracy, which is the property the tests pin.

Pure numpy/stdlib: no scikit-learn in the tested path.
"""

from __future__ import annotations

import numpy as np

from sentiment.clean import tokenize


def bag_of_words(
    docs: list[str], vocab: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Build an integer bag-of-words count matrix and a sorted vocabulary.

    Parameters
    ----------
    docs:
        A list of raw document strings, each tokenised with
        :func:`sentiment.clean.tokenize`.
    vocab:
        If given, use exactly this vocabulary (and column order); tokens not in
        it are ignored. This is how a *test* corpus is encoded against the
        vocabulary learned at ``fit`` time. If ``None`` (default) the vocabulary
        is the sorted set of tokens seen in ``docs``.

    Returns
    -------
    matrix : numpy.ndarray
        An ``(N, V)`` float array of raw token counts.
    vocab : list[str]
        The vocabulary; ``vocab[j]`` names column ``j``.

    Raises
    ------
    ValueError
        If ``docs`` is empty.

    Examples
    --------
    >>> X, vocab = bag_of_words(["good good bad", "bad"])
    >>> vocab
    ['bad', 'good']
    >>> X.tolist()
    [[1.0, 2.0], [1.0, 0.0]]
    """
    if not docs:
        raise ValueError("bag_of_words requires at least one document.")

    tokenized = [tokenize(d) for d in docs]
    if vocab is None:
        vocab = sorted({tok for toks in tokenized for tok in toks})
    index = {tok: j for j, tok in enumerate(vocab)}

    matrix = np.zeros((len(docs), len(vocab)), dtype=float)
    for i, toks in enumerate(tokenized):
        for tok in toks:
            j = index.get(tok)
            if j is not None:
                matrix[i, j] += 1.0
    return matrix, vocab


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid, elementwise."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    r"""Binary logistic-regression classifier fit by gradient descent (pure numpy).

    The model is :math:`\sigma(Xw + b)`; ``fit`` minimises the mean binary
    cross-entropy (optionally with an L2 penalty on ``w``) by full-batch gradient
    descent. It is the trained, learned-from-data alternative to the fixed
    lexicon scorer.

    Parameters
    ----------
    lr:
        Learning rate (gradient-descent step size).
    n_iters:
        Number of gradient-descent iterations.
    l2:
        L2 regularisation strength on the weights (not the bias). ``0`` disables
        it. Small values keep the fit well-behaved without preventing a
        separable set from reaching 100% train accuracy.

    Attributes
    ----------
    weights_ : numpy.ndarray
        The fitted coefficient vector, shape ``(V,)``.
    bias_ : float
        The fitted intercept.
    loss_history_ : list[float]
        Mean cross-entropy after each iteration (monotone non-increasing for a
        small enough ``lr``); handy for diagnostics and tests.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[2.0, 0.0], [0.0, 2.0]])
    >>> y = np.array([1, 0])
    >>> clf = LogisticRegression(lr=0.5, n_iters=500).fit(X, y)
    >>> clf.predict(X).tolist()
    [1, 0]
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 1000, l2: float = 0.0) -> None:
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.l2 = float(l2)
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Fit the model to feature matrix ``X`` and binary labels ``y``.

        Parameters
        ----------
        X:
            ``(N, V)`` numeric feature matrix (e.g. from :func:`bag_of_words`).
        y:
            Length-``N`` binary labels in ``{0, 1}``.

        Returns
        -------
        LogisticRegression
            ``self`` (fitted), to allow chaining.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, lengths disagree, or ``X`` is empty.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
        if X.shape[0] == 0:
            raise ValueError("fit requires at least one sample.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n, v = X.shape
        self.weights_ = np.zeros(v, dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            p = _sigmoid(X @ self.weights_ + self.bias_)
            error = p - y  # gradient of cross-entropy wrt logits
            grad_w = (X.T @ error) / n + self.l2 * self.weights_
            grad_b = float(error.mean())
            self.weights_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b

            # Mean binary cross-entropy (clipped for log stability).
            eps = 1e-12
            pc = np.clip(p, eps, 1.0 - eps)
            loss = float(-(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)).mean())
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(y = 1)`` for each row of ``X``.

        Parameters
        ----------
        X:
            ``(N, V)`` feature matrix with the same column meaning as at ``fit``.

        Returns
        -------
        numpy.ndarray
            Length-``N`` array of probabilities in ``[0, 1]``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.weights_ is None:
            raise RuntimeError("LogisticRegression must be fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard ``{0, 1}`` labels by thresholding :meth:`predict_proba`.

        Parameters
        ----------
        X:
            ``(N, V)`` feature matrix.
        threshold:
            Probabilities ``>= threshold`` map to class ``1``.

        Returns
        -------
        numpy.ndarray
            Length-``N`` integer array of predicted labels.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
