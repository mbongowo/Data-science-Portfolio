"""Unsupervised anomaly detectors over the event-count matrix (pure numpy).

Two cheap, transparent detectors that need no labels and no third-party ML:

* :func:`pca_reconstruction_error` — project each session onto the top-``k``
  principal components of the (centred) event-count matrix and measure how much
  is lost. Sessions that do not lie in the dominant subspace (rare event mixes)
  have a large reconstruction error. This is the classic PCA log-anomaly
  detector (Xu et al., 2009) in its smallest honest form.
* :func:`mahalanobis_scores` — score each session by its (squared) Mahalanobis
  distance from the column mean under a pseudo-inverse of the covariance. This
  is scale- and correlation-aware: a session is anomalous if its event mix is
  far from the centre *after* whitening, even when no single count is extreme.
  The pseudo-inverse keeps it stable when features are collinear or constant.
* :func:`zscore_anomalies` — flag scores that sit many standard deviations from
  the mean. A blunt instrument, useful as a baseline or on a single derived
  score.

:func:`flag` turns a vector of errors into a boolean mask by thresholding at a
quantile; :func:`mahalanobis_threshold` is the matching helper for Mahalanobis
scores. The heavier sklearn IsolationForest detector lives in ``spark_pipeline``
behind a lazy import, so this module stays dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def pca_reconstruction_error(X: ArrayLike, k: int) -> NDArray[np.float64]:
    r"""Per-row L2 reconstruction error after a rank-``k`` PCA projection.

    The matrix is mean-centred by column, decomposed with an SVD, and each row
    is projected onto the top-``k`` right singular vectors (principal
    directions). The error is the Euclidean norm of the residual

    .. math::

        e_i = \lVert (x_i - \bar{x}) - (x_i - \bar{x}) V_k V_k^\top \rVert_2 ,

    where :math:`V_k` holds the top-``k`` principal directions as columns. When
    ``k`` is at least the rank of the centred matrix, the projection is exact and
    every error is ~0.

    Parameters
    ----------
    X:
        ``(n_sessions, n_templates)`` event-count matrix.
    k:
        Number of principal components to keep (clamped to ``[0, n_features]``).

    Returns
    -------
    numpy.ndarray
        Length-``n_sessions`` array of non-negative reconstruction errors.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D or ``k`` is negative.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    if k < 0:
        raise ValueError("k must be non-negative.")

    n_features = A.shape[1]
    k = min(k, n_features)

    mean = A.mean(axis=0, keepdims=True)
    centered = A - mean

    if k == 0:
        return np.linalg.norm(centered, axis=1)

    # Right singular vectors are the principal directions (rows of Vt).
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]  # (k, n_features)

    projected = centered @ components.T @ components
    residual = centered - projected
    return np.linalg.norm(residual, axis=1)


def mahalanobis_scores(X: ArrayLike) -> NDArray[np.float64]:
    r"""Per-row squared Mahalanobis distance from the column mean.

    Each row is whitened by the (pseudo-inverted) sample covariance, so a
    session is scored by how far its event mix sits from the centre *after*
    accounting for the spread and correlation of the templates:

    .. math::

        d_i^2 = (x_i - \bar{x})\, \Sigma^{+} \,(x_i - \bar{x})^\top ,

    where :math:`\Sigma` is the (population, ``ddof=0``) covariance of the rows
    and :math:`\Sigma^{+}` its Moore-Penrose pseudo-inverse. The pseudo-inverse
    is what makes this robust: a constant (zero-variance) column or two
    perfectly collinear columns make :math:`\Sigma` singular, and a plain
    inverse would blow up; the pseudo-inverse simply ignores those null
    directions instead.

    Parameters
    ----------
    X:
        ``(n_sessions, n_templates)`` event-count matrix.

    Returns
    -------
    numpy.ndarray
        Length-``n_sessions`` array of non-negative squared distances.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D.

    Notes
    -----
    A degenerate direction contributes nothing: if every row shares the same
    value in some column, that column is in the null space of :math:`\Sigma` and
    does not affect any score. With a single row the covariance is all-zeros and
    every score is ``0``.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError("X must be a 2-D array.")

    mean = A.mean(axis=0, keepdims=True)
    centered = A - mean

    # Population covariance (ddof=0); rowvar=False -> columns are the variables.
    cov = np.cov(A, rowvar=False, ddof=0)
    cov = np.atleast_2d(cov)
    cov_pinv = np.linalg.pinv(cov)

    # d_i^2 = row_i @ Sigma^+ @ row_i^T, vectorised over rows.
    scores = np.einsum("ij,jk,ik->i", centered, cov_pinv, centered)
    # Clip tiny negative values from floating-point round-off.
    return np.maximum(scores, 0.0)


def mahalanobis_threshold(scores: ArrayLike, quantile: float) -> NDArray[np.bool_]:
    """Flag Mahalanobis scores strictly above the given ``quantile``.

    A thin convenience wrapper around :func:`flag` so the Mahalanobis detector
    reads symmetrically with the PCA one. ``0.95`` flags the top ~5% of sessions
    by distance from the centre.

    Parameters
    ----------
    scores:
        Length-``n`` array of (squared) Mahalanobis distances.
    quantile:
        Quantile in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the score exceeds the quantile cut.

    Raises
    ------
    ValueError
        If ``quantile`` is outside ``[0, 1]``.
    """
    return flag(scores, quantile)


def zscore_anomalies(scores: ArrayLike, z: float) -> NDArray[np.bool_]:
    """Flag scores more than ``z`` standard deviations above the mean.

    The test is one-sided on the standardised score ``(s - mean) / std > z``, so
    only unusually *high* scores are flagged. A zero-variance input flags
    nothing (there is no spread to be anomalous against).

    Parameters
    ----------
    scores:
        Length-``n`` array of per-session scores.
    z:
        Threshold in standard deviations.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the score exceeds the threshold.
    """
    s = np.asarray(scores, dtype=float).ravel()
    std = s.std()
    if std == 0.0:
        return np.zeros(s.shape, dtype=bool)
    return (s - s.mean()) / std > z


def flag(errors: ArrayLike, quantile: float) -> NDArray[np.bool_]:
    """Flag errors strictly above the given ``quantile`` of the error vector.

    Parameters
    ----------
    errors:
        Length-``n`` array of non-negative anomaly scores / reconstruction
        errors.
    quantile:
        Quantile in ``[0, 1]``. ``0.99`` flags the top ~1% of sessions.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where ``error`` exceeds the quantile cut.

    Raises
    ------
    ValueError
        If ``quantile`` is outside ``[0, 1]``.
    """
    e = np.asarray(errors, dtype=float).ravel()
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    cut = float(np.quantile(e, quantile))
    return e > cut
