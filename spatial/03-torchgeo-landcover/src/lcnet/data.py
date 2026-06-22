"""Pure-numpy data utilities: splitting, statistics, and the imagery bridge.

These helpers depend only on :mod:`numpy` and turn raw multispectral patches
into the feature vectors the :class:`lcnet.classifier.SoftmaxClassifier`
consumes, plus the deterministic, class-balanced split that keeps evaluation
honest. No torch / rasterio / torchgeo here, so everything is CI-testable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["stratified_split", "band_stats", "patch_features"]


def stratified_split(
    y: np.ndarray,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic class-balanced train / val / test index split.

    Every class is split *within itself* by ``fractions``, so each split holds a
    representative share of every land-cover class (no class is missing from a
    split as long as it has at least three samples). The result is deterministic
    given ``seed``.

    Parameters
    ----------
    y : numpy.ndarray
        Integer class labels, shape ``(n_samples,)``.
    fractions : tuple of float, optional
        ``(train, val, test)`` fractions; must be positive and sum to ``1.0``.
    seed : int, optional
        Seed for the per-class shuffle.

    Returns
    -------
    tuple of numpy.ndarray
        ``(train_idx, val_idx, test_idx)`` — disjoint integer index arrays whose
        union is every sample. Each array is sorted.

    Raises
    ------
    ValueError
        If ``fractions`` does not have three positive entries summing to 1.
    """
    y = np.asarray(y).reshape(-1)
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be three positive numbers")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    f_train, f_val, _ = fractions
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = idx.size
        # Guarantee at least one sample in train and val when the class has >= 3
        # members, so no split silently loses a class to rounding.
        n_train = int(np.floor(n * f_train))
        n_val = int(np.floor(n * f_val))
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])

    train_idx = (
        np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    )
    val_idx = (
        np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    )
    test_idx = (
        np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    )
    return train_idx, val_idx, test_idx


def band_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ``(mean, std)`` of a feature matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, std)``, each shape ``(n_features,)``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D (n_samples, n_features) array")
    return X.mean(axis=0), X.std(axis=0)


def patch_features(patch: np.ndarray) -> np.ndarray:
    """Reduce a multispectral patch to a per-band ``[mean, std]`` feature vector.

    This is the bridge from imagery to the classifier: a ``(C, H, W)`` patch (C
    spectral bands) collapses to a length-``2C`` vector ``[mean_0, std_0,
    mean_1, std_1, ...]``. The per-band mean captures the spectral signature
    that separates land-cover classes (vegetation vs. water vs. built-up) and
    the per-band std captures within-patch texture.

    Parameters
    ----------
    patch : numpy.ndarray
        Multispectral patch, shape ``(C, H, W)``.

    Returns
    -------
    numpy.ndarray
        Feature vector of length ``2 * C``.

    Raises
    ------
    ValueError
        If ``patch`` is not 3-D.
    """
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim != 3:
        raise ValueError("patch must be a 3-D (C, H, W) array")
    c = patch.shape[0]
    flat = patch.reshape(c, -1)
    means = flat.mean(axis=1)
    stds = flat.std(axis=1)
    # Interleave [mean_0, std_0, mean_1, std_1, ...].
    feats = np.empty(2 * c, dtype=np.float64)
    feats[0::2] = means
    feats[1::2] = stds
    return feats
