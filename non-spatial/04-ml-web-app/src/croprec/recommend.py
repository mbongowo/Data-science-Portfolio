"""Turn a single soil-and-climate reading into a ranked crop recommendation.

This is the exact flow the Streamlit app runs on submit:

1. The app trains the :class:`~croprec.model.SoftmaxClassifier` once on the
   standardised training split, keeping the training ``feature_means`` and
   ``feature_stds`` (the z-score statistics) and the ``classes`` label array.
2. A user's input is a dict of the seven features. :func:`recommend` arranges it
   in canonical feature order, standardises it with the *training* statistics
   (never re-estimated from one row), runs ``predict_proba``, and returns the
   crops sorted by descending probability.

Keeping the standardisation statistics with the model is what makes a single new
sample comparable to the training distribution.
"""

from __future__ import annotations

import numpy as np

from croprec.data import FEATURE_COLUMNS
from croprec.model import SoftmaxClassifier, standardize


def recommend(
    model: SoftmaxClassifier,
    classes,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    sample: dict,
    feature_cols: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Recommend crops for one input ``sample``.

    Parameters
    ----------
    model:
        A fitted :class:`SoftmaxClassifier`.
    classes:
        The crop label array aligned with the model's ``classes_`` columns.
    feature_means, feature_stds:
        The training-set z-score statistics (from :func:`standardize`).
    sample:
        Dict mapping each feature name to its value.
    feature_cols:
        Feature order; defaults to :data:`croprec.data.FEATURE_COLUMNS`.

    Returns
    -------
    list[tuple[str, float]]
        ``(crop, probability)`` pairs sorted by descending probability; the
        probabilities sum to ~1.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
    missing = [c for c in cols if c not in sample]
    if missing:
        raise ValueError(f"sample is missing features: {missing}")

    x = np.array([[float(sample[c]) for c in cols]], dtype=float)
    x_scaled, _, _ = standardize(x, mean=feature_means, std=feature_stds)
    proba = model.predict_proba(x_scaled)[0]

    classes = np.asarray(classes, dtype=object)
    order = np.argsort(proba)[::-1]
    return [(str(classes[i]), float(proba[i])) for i in order]
