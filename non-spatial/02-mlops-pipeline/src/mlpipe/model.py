"""A pure-numpy logistic-regression classifier.

The runnable, CI-tested model for the rain-day task. It is deliberately small
and transparent — sigmoid link, numerically stable cross-entropy loss, optional
L2 penalty, batch gradient descent — so its behaviour is fully reproducible and
the maths is checkable. The heavier sklearn / gradient-boosted path is an opt-in
swap behind the tracking wrapper; nothing here needs anything beyond numpy.

:func:`standardize` z-scores a feature matrix (and can reuse stored train
statistics on new data, which is what avoids leakage at serving time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def standardize(
    X: ArrayLike,
    mean: ArrayLike | None = None,
    std: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Z-score columns of ``X`` to zero mean / unit standard deviation.

    Pass ``mean`` and ``std`` to reuse statistics computed on the training set
    (the correct thing to do for validation / serving data — fit the scaler on
    train only). When they are ``None`` they are estimated from ``X``. Columns
    with zero variance are divided by 1 instead of 0.

    Parameters
    ----------
    X:
        2-D feature matrix ``(n_samples, n_features)``.
    mean, std:
        Optional per-column statistics to apply. If omitted, computed from ``X``.

    Returns
    -------
    tuple
        ``(X_scaled, mean, std)`` — the scaled matrix and the statistics used
        (so they can be stored and re-applied).
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("standardize expects a 2-D feature matrix.")
    mu = np.asarray(mean, dtype=float) if mean is not None else arr.mean(axis=0)
    sd = np.asarray(std, dtype=float) if std is not None else arr.std(axis=0)
    safe_sd = np.where(sd == 0.0, 1.0, sd)
    return (arr - mu) / safe_sd, mu, safe_sd


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    """Binary logistic regression trained by batch gradient descent.

    Parameters
    ----------
    lr:
        Learning rate.
    epochs:
        Number of full-batch gradient steps.
    l2:
        L2 regularisation strength (the bias term is not penalised).
    seed:
        Seed for the (tiny) weight initialisation.

    Attributes
    ----------
    weights_, bias_:
        Learned parameters, available after :meth:`fit`.
    loss_history_:
        Mean regularised cross-entropy after each epoch (monotone non-increasing
        for a small enough learning rate), available after :meth:`fit`.
    """

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 500,
        l2: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.weights_: NDArray[np.float64] | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        """Fit the model in place and return ``self``.

        Parameters
        ----------
        X:
            Feature matrix ``(n_samples, n_features)``.
        y:
            Binary labels in ``{0, 1}``, length ``n_samples``.

        Raises
        ------
        ValueError
            If shapes disagree or the data is empty.
        """
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float).ravel()
        if Xa.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix.")
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError(
                f"X has {Xa.shape[0]} rows but y has {ya.shape[0]} labels."
            )
        if Xa.shape[0] == 0:
            raise ValueError("Cannot fit on empty data.")

        n, d = Xa.shape
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, size=d)
        b = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            z = Xa @ w + b
            p = _sigmoid(z)
            error = p - ya
            grad_w = Xa.T @ error / n + self.l2 * w
            grad_b = float(np.mean(error))
            w -= self.lr * grad_w
            b -= self.lr * grad_b

            # Stable cross-entropy via log1p; add the L2 penalty on the weights.
            log_loss = float(np.mean(np.logaddexp(0.0, z) - ya * z))
            reg = 0.5 * self.l2 * float(np.dot(w, w))
            self.loss_history_.append(log_loss + reg)

        self.weights_ = w
        self.bias_ = b
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the probability of the positive class for each row, in ``[0, 1]``."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted; call fit first.")
        Xa = np.asarray(X, dtype=float)
        return _sigmoid(Xa @ self.weights_ + self.bias_)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray[np.int_]:
        """Return 0/1 predictions by thresholding :meth:`predict_proba`."""
        return (self.predict_proba(X) >= threshold).astype(int)
