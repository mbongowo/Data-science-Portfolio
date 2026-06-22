"""Event-count features: turn template-id sequences into a session-by-event matrix.

After templating, each session (an HDFS block, say) is a sequence of template
ids. The standard feature representation for log anomaly detection is the
**event-count matrix**: one row per session, one column per template, each cell
the number of times that template fired in that session. PCA, z-score, and
IsolationForest detectors all operate on this matrix.

Pure numpy/stdlib; no third-party dependency beyond numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def event_count_matrix(
    session_to_template_ids: Mapping[object, Iterable[int]],
    n_templates: int,
) -> NDArray[np.float64]:
    """Build an event-count matrix from per-session template-id sequences.

    Parameters
    ----------
    session_to_template_ids:
        Mapping from a session key to its sequence of template ids. Iteration
        order of the mapping fixes the row order of the output (dicts preserve
        insertion order in Python 3.7+).
    n_templates:
        Total number of distinct templates; the number of columns. Every id in
        the sequences must be in ``range(n_templates)``.

    Returns
    -------
    numpy.ndarray
        A ``(n_sessions, n_templates)`` float array. Cell ``(i, j)`` is the
        number of times template ``j`` occurred in session ``i``.

    Raises
    ------
    ValueError
        If ``n_templates`` is negative, or a template id falls outside
        ``range(n_templates)``.

    Examples
    --------
    Two sessions over three templates:

    >>> m = event_count_matrix({"a": [0, 0, 1], "b": [2, 1, 1]}, 3)
    >>> m.tolist()
    [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]
    """
    if n_templates < 0:
        raise ValueError("n_templates must be non-negative.")

    n_sessions = len(session_to_template_ids)
    matrix = np.zeros((n_sessions, n_templates), dtype=float)

    for row, ids in enumerate(session_to_template_ids.values()):
        for tid in ids:
            if not 0 <= tid < n_templates:
                raise ValueError(f"Template id {tid} out of range [0, {n_templates}).")
            matrix[row, tid] += 1.0
    return matrix


def template_idf(
    session_template_ids: Iterable[Iterable[int]],
    n_templates: int,
) -> NDArray[np.float64]:
    r"""Inverse-document-frequency weight per template (rarity weight).

    Treating each session as a "document" and each template as a "term", a
    template that appears in many sessions is common boilerplate and carries
    little signal; one that appears in only a handful is rare and informative.
    The smoothed IDF makes this concrete:

    .. math::

        \mathrm{idf}_j = \ln\!\left(\frac{1 + N}{1 + \mathrm{df}_j}\right) + 1 ,

    where :math:`N` is the number of sessions and :math:`\mathrm{df}_j` the
    number of sessions in which template ``j`` appears at least once. The ``+1``
    smoothing (as in scikit-learn's ``TfidfTransformer``) keeps a template that
    appears in *every* session at a positive weight of ``1.0`` rather than
    collapsing it to ``0``, and avoids a division by zero for an unseen template.
    Higher weight == rarer == more worth attending to.

    Parameters
    ----------
    session_template_ids:
        Iterable of per-session template-id sequences (the same structure the
        event-count matrix is built from).
    n_templates:
        Number of distinct templates; the length of the returned vector.

    Returns
    -------
    numpy.ndarray
        Length-``n_templates`` array of non-negative IDF weights.

    Raises
    ------
    ValueError
        If ``n_templates`` is negative, or a template id is out of range.
    """
    if n_templates < 0:
        raise ValueError("n_templates must be non-negative.")

    doc_freq = np.zeros(n_templates, dtype=float)
    n_sessions = 0
    for ids in session_template_ids:
        n_sessions += 1
        seen: set[int] = set()
        for tid in ids:
            if not 0 <= tid < n_templates:
                raise ValueError(f"Template id {tid} out of range [0, {n_templates}).")
            seen.add(int(tid))
        for tid in seen:
            doc_freq[tid] += 1.0

    return np.log((1.0 + n_sessions) / (1.0 + doc_freq)) + 1.0


def session_rarity(
    event_counts: ArrayLike,
    idf: ArrayLike,
) -> NDArray[np.float64]:
    """Per-session rarity score: event counts weighted by template IDF.

    Each session's score is the dot product of its event-count row with the
    template IDF weights, ``score_i = sum_j count_ij * idf_j``. A session built
    from common templates scores low; one that fires rare templates (or fires
    them repeatedly) scores high. This is a cheap, fully transparent anomaly
    signal that complements the PCA / Mahalanobis detectors and needs no labels.

    Parameters
    ----------
    event_counts:
        ``(n_sessions, n_templates)`` event-count matrix (see
        :func:`event_count_matrix`).
    idf:
        Length-``n_templates`` template IDF weights (see :func:`template_idf`).

    Returns
    -------
    numpy.ndarray
        Length-``n_sessions`` array of non-negative rarity scores.

    Raises
    ------
    ValueError
        If the matrix is not 2-D or its column count does not match ``idf``.
    """
    X = np.asarray(event_counts, dtype=float)
    w = np.asarray(idf, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("event_counts must be a 2-D array.")
    if X.shape[1] != w.shape[0]:
        raise ValueError("event_counts columns must match the length of idf.")
    return X @ w


def count_invariants(
    event_count_matrix: ArrayLike,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> NDArray[np.bool_]:
    """Flag sessions whose template counts fall outside a learned normal band.

    A simple, interpretable "invariant" detector: for each template column,
    learn a normal range as the ``[lower_quantile, upper_quantile]`` band of the
    counts across all sessions. A session is flagged ``True`` if *any* of its
    template counts falls outside that column's band — i.e. it fires some
    template far more or far less often than the bulk of sessions do.

    Unlike PCA / Mahalanobis (which score a whole-vector distance), this is a
    per-template rule and says *which* template broke its invariant is easy to
    recover by comparing the row to the bands.

    Parameters
    ----------
    event_count_matrix:
        ``(n_sessions, n_templates)`` event-count matrix.
    lower_quantile, upper_quantile:
        The normal band per column, in ``[0, 1]`` with
        ``lower_quantile <= upper_quantile``. Defaults to the central 90%.

    Returns
    -------
    numpy.ndarray
        Length-``n_sessions`` boolean mask, ``True`` where the session violates
        at least one column's band.

    Raises
    ------
    ValueError
        If the matrix is not 2-D or the quantiles are invalid.
    """
    X = np.asarray(event_count_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("event_count_matrix must be a 2-D array.")
    if not 0.0 <= lower_quantile <= upper_quantile <= 1.0:
        raise ValueError("need 0 <= lower_quantile <= upper_quantile <= 1.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=bool)

    lo = np.quantile(X, lower_quantile, axis=0)
    hi = np.quantile(X, upper_quantile, axis=0)
    out_of_band = (lo > X) | (hi < X)  # (n_sessions, n_templates)
    return out_of_band.any(axis=1)
