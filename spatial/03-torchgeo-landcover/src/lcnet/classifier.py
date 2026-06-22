"""Pure-numpy softmax (multinomial logistic regression) classifier.

A from-scratch, fully tested baseline for whole-patch land-cover classification:
batch gradient descent on the cross-entropy loss of a numerically stable softmax,
with optional L2 regularisation. It depends only on :mod:`numpy`, so it trains
and tests without torch / a GPU and serves as the "from-scratch" arm of the
transfer-learning comparison documented in the README.

It is a *linear* model on the feature vectors produced by
:func:`lcnet.data.patch_features`, so it reaches ~100% training accuracy on
linearly separable classes and gives an honest, reproducible floor against which
the TorchGeo ResNet fine-tune (see :mod:`lcnet.train`) is measured.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SoftmaxClassifier", "standardize"]


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance per column.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    mean, std : numpy.ndarray, optional
        Pre-fitted per-feature mean / std. If ``None`` they are estimated from
        ``X``; pass the *training* statistics here to transform a validation or
        test split with the same scaling.

    Returns
    -------
    tuple
        ``(X_std, mean, std)``. Zero-variance columns are scaled by ``1.0``
        instead of dividing by zero, so a constant feature maps to all zeros.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D (n_samples, n_features) array")
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    safe_std = np.where(std > 0, std, 1.0)
    X_std = (X - mean) / safe_std
    return X_std, mean, std


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """Multinomial logistic regression trained by batch gradient descent.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If ``None`` it is inferred from ``y`` at :meth:`fit`
        time as ``y.max() + 1``.

    Attributes
    ----------
    W : numpy.ndarray
        Weight matrix, shape ``(n_features, num_classes)``.
    b : numpy.ndarray
        Bias vector, shape ``(num_classes,)``.
    loss_history : list of float
        Mean cross-entropy (plus L2 penalty) after each epoch.
    """

    def __init__(self, num_classes: int | None = None) -> None:
        self.num_classes = num_classes
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.5,
        epochs: int = 300,
        l2: float = 0.0,
        seed: int = 0,
    ) -> SoftmaxClassifier:
        """Fit the classifier with full-batch gradient descent.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix, shape ``(n_samples, n_features)``.
        y : numpy.ndarray
            Integer class labels, shape ``(n_samples,)``.
        lr : float, optional
            Learning rate (step size).
        epochs : int, optional
            Number of full passes over the data.
        l2 : float, optional
            L2 regularisation strength on the weights (not the bias).
        seed : int, optional
            Seed for the small random weight initialisation, so a given dataset
            and seed always produce the same model and loss curve.

        Returns
        -------
        SoftmaxClassifier
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D or its rows do not match ``len(y)``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1).astype(np.int64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D (n_samples, n_features) array")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"row mismatch: X {X.shape[0]} vs y {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        n_samples, n_features = X.shape
        num_classes = self.num_classes
        if num_classes is None:
            num_classes = int(y.max()) + 1
            self.num_classes = num_classes
        if y.min() < 0 or y.max() >= num_classes:
            raise ValueError("labels must lie in range(num_classes)")

        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, size=(n_features, num_classes))
        self.b = np.zeros(num_classes, dtype=np.float64)

        # One-hot targets for the cross-entropy gradient.
        onehot = np.zeros((n_samples, num_classes), dtype=np.float64)
        onehot[np.arange(n_samples), y] = 1.0

        self.loss_history = []
        for _ in range(epochs):
            logits = X @ self.W + self.b
            proba = _softmax(logits)

            # Cross-entropy loss (+ L2). Clip to avoid log(0).
            log_likelihood = -np.log(
                np.clip(proba[np.arange(n_samples), y], 1e-12, 1.0)
            )
            loss = float(np.mean(log_likelihood))
            loss += 0.5 * l2 * float(np.sum(self.W**2)) / n_samples
            self.loss_history.append(loss)

            # Gradients of mean cross-entropy w.r.t. W and b.
            grad_logits = (proba - onehot) / n_samples
            grad_W = X.T @ grad_logits + (l2 / n_samples) * self.W
            grad_b = grad_logits.sum(axis=0)

            self.W -= lr * grad_W
            self.b -= lr * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class-probability matrix, shape ``(n_samples, num_classes)``.

        Each row is a valid probability distribution (non-negative, sums to 1).
        """
        if self.W is None or self.b is None:
            raise ValueError("classifier is not fitted; call fit() first")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D (n_samples, n_features) array")
        return _softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class index per sample, shape ``(n_samples,)``."""
        return np.argmax(self.predict_proba(X), axis=1)
