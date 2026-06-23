"""Pure-numpy multinomial logistic regression (softmax classifier).

The classifier is plain batch gradient descent over the multinomial
cross-entropy loss with L2 regularisation. Everything is hand-derivable:

* ``softmax(Z)`` is computed in a numerically stable way by subtracting each
  row's max before exponentiating, so large logits do not overflow.
* The gradient of the mean cross-entropy plus ``l2 * ||W||^2`` with respect to
  the weights is ``X.T @ (P - Y_onehot) / n + 2 * l2 * W`` (the bias row is not
  penalised), which is what :meth:`SoftmaxClassifier.fit` descends.

It has no dependency beyond numpy and trains in well under a second on the small
tabular datasets used here, reaching ~1.0 accuracy on linearly separable
classes. :func:`standardize` is the companion z-score scaler the app fits on the
training split and reuses for every prediction.
"""

from __future__ import annotations

import numpy as np


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the columns of ``X``.

    If ``mean`` / ``std`` are not given they are estimated from ``X`` (this is
    the training call); pass the training statistics back in to transform a new
    split with the *same* scaling. A zero std (a constant column) is replaced by
    1.0 so the column maps to all-zeros instead of dividing by zero.

    Returns ``(X_scaled, mean, std)``.
    """
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.asarray(std, dtype=float).copy()
    std[std == 0.0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled, np.asarray(mean, dtype=float), std


def _softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """Multinomial logistic regression trained by batch gradient descent.

    Attributes
    ----------
    W:
        Weight matrix of shape ``(n_features + 1, n_classes)``; the last row is
        the bias. ``None`` until :meth:`fit` is called.
    classes_:
        Sorted array of the integer class labels seen in training.
    loss_history:
        Mean training loss (cross-entropy + L2) recorded once per epoch.
    """

    def __init__(self) -> None:
        self.W: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.loss_history: list[float] = []

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.5,
        epochs: int = 400,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> SoftmaxClassifier:
        """Fit the classifier.

        Parameters
        ----------
        X:
            Feature matrix ``(n_samples, n_features)`` — standardise first.
        y:
            Integer class labels ``(n_samples,)``.
        lr:
            Gradient-descent step size.
        epochs:
            Number of full-batch passes.
        l2:
            L2 penalty on the (non-bias) weights.
        seed:
            Seed for the small random weight initialisation, so a given
            ``(X, y, lr, epochs, l2, seed)`` is fully reproducible.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            raise ValueError("need at least two classes to fit")

        # Map labels to 0..n_classes-1 columns of the one-hot target.
        class_to_col = {c: i for i, c in enumerate(self.classes_)}
        cols = np.array([class_to_col[c] for c in y])
        n = X.shape[0]
        Y = np.zeros((n, n_classes))
        Y[np.arange(n), cols] = 1.0

        Xb = self._add_bias(X)
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((Xb.shape[1], n_classes))
        self.loss_history = []

        for _ in range(int(epochs)):
            P = _softmax(Xb @ self.W)
            # Mean cross-entropy + L2 on the weight rows (not the bias row).
            eps = 1e-12
            ce = -np.sum(Y * np.log(P + eps)) / n
            reg = l2 * np.sum(self.W[:-1] ** 2)
            self.loss_history.append(float(ce + reg))

            grad = Xb.T @ (P - Y) / n
            grad[:-1] += 2.0 * l2 * self.W[:-1]
            self.W -= lr * grad

        return self

    def _check_fitted(self) -> None:
        if self.W is None or self.classes_ is None:
            raise RuntimeError("classifier is not fitted; call fit() first")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities ``(n_samples, n_classes)`` (rows sum to 1)."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return _softmax(self._add_bias(X) @ self.W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label per row."""
        self._check_fitted()
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def top_k(self, X: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``k`` highest-probability classes per row.

        Returns ``(classes, probabilities)``, each of shape ``(n_samples, k)``,
        with columns ordered by descending probability. ``k`` is clipped to the
        number of classes.
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        k = int(min(k, proba.shape[1]))
        if k < 1:
            raise ValueError("k must be >= 1")
        # argsort ascending then take the last k reversed -> descending.
        order = np.argsort(proba, axis=1)[:, ::-1][:, :k]
        top_classes = self.classes_[order]
        rows = np.arange(proba.shape[0])[:, None]
        top_proba = proba[rows, order]
        return top_classes, top_proba
