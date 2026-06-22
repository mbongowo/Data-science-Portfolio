"""Unsupervised anomaly detectors over the event-count matrix (pure numpy).

Two cheap, transparent detectors that need no labels and no third-party ML:

* :func:`pca_reconstruction_error` — project each session onto the top-``k``
  principal components of the (centred) event-count matrix and measure how much
  is lost. Sessions that do not lie in the dominant subspace (rare event mixes)
  have a large reconstruction error. This is the classic PCA log-anomaly
  detector (Xu et al., 2009) in its smallest honest form.
* :func:`zscore_anomalies` — flag scores that sit many standard deviations from
  the mean. A blunt instrument, useful as a baseline or on a single derived
  score.

:func:`flag` turns a vector of errors into a boolean mask by thresholding at a
quantile. The heavier sklearn IsolationForest detector lives in
``spark_pipeline`` behind a lazy import, so this module stays dependency-free.
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
