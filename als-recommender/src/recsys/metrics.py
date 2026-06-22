"""Rating-error and ranking-quality metrics for recommender evaluation.

This is a pure-numpy / pure-Python reference layer with no third-party
dependency beyond numpy, so it is always importable and is the basis of the
known-answer unit tests.

Two families of metric live here:

* **Rating error** — :func:`rmse` for the regression view (how close are the
  predicted ratings to the held-out ratings).
* **Ranking quality** — :func:`precision_at_k`, :func:`recall_at_k`, and
  :func:`ndcg_at_k` for the top-N view (is the ordered list of recommended
  items any good). Ranking metrics are what actually matter for a recommender;
  a model can have a good RMSE and still rank badly.

Interpretation notes (do not skip these):

* These are *offline* metrics computed against logged interactions. They measure
  agreement with the past, not the value of a recommendation shown to a user.
* Precision@K / Recall@K ignore the order within the top-K; NDCG@K does not.
* All three depend on the cut-off ``k`` and on how "relevant" is defined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Root-mean-square error between true and predicted ratings.

    .. math::

        \mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

    Parameters
    ----------
    y_true, y_pred:
        Equal-length arrays of observed and predicted ratings.

    Returns
    -------
    float
        The RMSE. Zero means perfect prediction; the units are rating units.

    Raises
    ------
    ValueError
        If the inputs are empty or of different length.
    """
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("RMSE needs at least one observation.")
    if a.shape != b.shape:
        raise ValueError(
            f"y_true and y_pred have different lengths: {a.size} vs {b.size}."
        )
    return float(np.sqrt(np.mean((a - b) ** 2)))


def precision_at_k(
    recommended: Sequence[object], relevant: set[object], k: int
) -> float:
    """Precision@K: fraction of the top-``k`` recommendations that are relevant.

    Parameters
    ----------
    recommended:
        Ordered list of recommended item ids, best first.
    relevant:
        Set of item ids that are actually relevant for this user.
    k:
        Cut-off. Only the first ``k`` recommended items are scored. The
        denominator is ``k`` (a short list still divides by ``k``), which is the
        usual convention.

    Returns
    -------
    float
        ``hits / k`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    top = recommended[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[object], relevant: set[object], k: int) -> float:
    """Recall@K: fraction of the relevant items that appear in the top-``k``.

    Parameters
    ----------
    recommended:
        Ordered list of recommended item ids, best first.
    relevant:
        Set of item ids that are actually relevant for this user.
    k:
        Cut-off. Only the first ``k`` recommended items are scored. The
        denominator is ``len(relevant)``.

    Returns
    -------
    float
        ``hits / len(relevant)`` in ``[0, 1]``. Returns ``0.0`` when the user has
        no relevant items (nothing could have been recalled).

    Raises
    ------
    ValueError
        If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: Sequence[object],
    relevance: Mapping[object, float],
    k: int,
) -> float:
    r"""Normalised Discounted Cumulative Gain at ``k``.

    Gain for an item is its graded relevance ``relevance[item]`` (0 if the item
    is absent from the mapping). The discount for the item at rank ``r`` (with
    ``r`` starting at 1) is ``1 / log2(r + 1)``, so the first item has discount
    ``1 / log2(2) = 1``.

    .. math::

        \mathrm{DCG@k} = \sum_{r=1}^{k} \frac{g_r}{\log_2(r + 1)},
        \qquad
        \mathrm{NDCG@k} = \frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}

    where ``IDCG@k`` is the DCG of the ideal ordering (relevances sorted
    descending). A perfect ranking gives ``NDCG@k = 1.0``.

    Parameters
    ----------
    recommended:
        Ordered list of recommended item ids, best first.
    relevance:
        Mapping from item id to graded relevance (non-negative). Items not in
        the mapping contribute zero gain.
    k:
        Cut-off.

    Returns
    -------
    float
        NDCG@k in ``[0, 1]``. Returns ``0.0`` when there is no attainable gain
        (IDCG is zero), i.e. the user has no relevant items.

    Raises
    ------
    ValueError
        If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    def _dcg(gains: Sequence[float]) -> float:
        return float(sum(g / np.log2(r + 1) for r, g in enumerate(gains[:k], start=1)))

    gains = [float(relevance.get(item, 0.0)) for item in recommended]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)

    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(gains) / idcg
