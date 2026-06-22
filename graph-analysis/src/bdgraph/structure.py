r"""Structural graph summaries: k-core, modularity, degree statistics.

Part of the **pure-numpy reference layer**: numpy + stdlib only, always
importable, and covered by hand-derived known-answer tests. These are the
structural diagnostics a reviewer expects to sit beside PageRank and triangle
counting: how cohesive each node's neighbourhood is (k-core), how good a
community partition is (modularity), and the basic shape of the degree
distribution.

* :func:`k_core_decomposition` returns the **core number** of every node: the
  largest ``k`` such that the node survives in the ``k``-core (the maximal
  subgraph in which every node has degree at least ``k``). It is computed by
  repeatedly peeling the lowest-degree node, an exact :math:`O(m)`-ish loop on
  the small graphs here. A clique ``K_m`` has every core number ``m-1``; a path
  has every core number ``1``.

* :func:`modularity` scores a partition against the configuration-model null:
  positive means the partition has more within-group edges than chance. Two
  disjoint cliques score high and positive; a meaningless single-group or random
  partition scores about zero (and a singleton partition is strongly negative).

* :func:`degree_stats` reports mean / max / min degree and the degree histogram.

Interpretation notes (do not skip these):

* The core number is a robustness measure, not a ranking of importance: a high
  core number means the node sits in a densely mutually-connected region.
* Modularity has a **resolution limit** and a known bias toward partitions of a
  particular scale; a higher modularity is not automatically a "better"
  partition, only a denser-within one relative to the null.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def _binary_undirected(adj: ArrayLike) -> NDArray[np.float64]:
    """Return a symmetric 0/1 copy of ``adj`` with a zero diagonal."""
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be a square 2-D matrix, got {a.shape}.")
    a = (a != 0).astype(float)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


def k_core_decomposition(adj: ArrayLike) -> NDArray[np.int_]:
    r"""Core number of every node by repeated minimum-degree peeling.

    The ``k``-core is the maximal subgraph in which every node has degree at
    least ``k``. A node's **core number** is the largest ``k`` whose ``k``-core
    still contains it. The standard peeling algorithm removes the current
    lowest-degree node, records its core number as the running maximum of the
    degrees seen at removal time, and decrements its neighbours' degrees.

    The graph is treated as **undirected and simple** (binarised, symmetrised,
    self-loops ignored).

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency. Binarised and symmetrised here.

    Returns
    -------
    numpy.ndarray
        Length-``n`` integer core number per node.

    Raises
    ------
    ValueError
        If ``adj`` is not square.

    Notes
    -----
    Hand-checkable: every node of a clique ``K_m`` has core number ``m - 1``;
    every node of a path or cycle has core number ``1`` (cycle: ``2``); an
    isolated node has core number ``0``.
    """
    a = _binary_undirected(adj)
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    degree = a.sum(axis=1).astype(int)
    neighbours = [set(np.flatnonzero(a[i] > 0).tolist()) for i in range(n)]
    core = np.zeros(n, dtype=int)
    removed = np.zeros(n, dtype=bool)
    level = 0

    for _ in range(n):
        # Lowest-degree node still present.
        masked = np.where(removed, np.iinfo(np.int64).max, degree)
        v = int(np.argmin(masked))
        level = max(level, int(degree[v]))
        core[v] = level
        removed[v] = True
        for w in neighbours[v]:
            if not removed[w] and degree[w] > 0:
                degree[w] -= 1

    return core


def modularity(adj: ArrayLike, labels: ArrayLike) -> float:
    r"""Newman-Girvan modularity of a partition on an undirected graph.

    Modularity compares the fraction of edges falling **inside** communities
    with the fraction expected under the configuration-model null (degrees
    preserved, edges otherwise random):

    .. math::

        Q = \frac{1}{2m} \sum_{ij}
            \left( A_{ij} - \frac{k_i k_j}{2m} \right)
            \,\delta(c_i, c_j),

    where ``A`` is the symmetric adjacency, ``k_i`` the degree of ``i``, ``m``
    the number of edges, and ``delta`` is 1 when ``i`` and ``j`` share a
    community. ``Q`` lies in ``[-1/2, 1)``; positive means denser-within than
    chance.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency. **Weighted** graphs are honoured (the
        symmetrised weights are used as ``A`` and ``2m`` is the total weight);
        the diagonal is ignored.
    labels:
        Length-``n`` community id per node (any hashable integers).

    Returns
    -------
    float
        The modularity ``Q``. Returns 0.0 for an edgeless graph.

    Raises
    ------
    ValueError
        If ``adj`` is not square or ``labels`` has the wrong length.

    Notes
    -----
    Hand-checkable: two disjoint triangles, each its own community, give
    ``Q = 0.5``; putting every node in one community gives ``Q = 0``.
    """
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be a square 2-D matrix, got {a.shape}.")
    n = a.shape[0]
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    a = np.maximum(a, a.T)  # symmetrise: undirected

    lab = np.asarray(labels).ravel()
    if lab.shape[0] != n:
        raise ValueError(f"labels must have length {n}, got {lab.shape[0]}.")

    k = a.sum(axis=1)  # weighted degree
    two_m = float(k.sum())  # = 2m
    if two_m == 0.0:
        return 0.0

    same = lab[:, None] == lab[None, :]
    expected = np.outer(k, k) / two_m
    q = float((a - expected)[same].sum()) / two_m
    return q


def degree_stats(adj: ArrayLike) -> dict[str, Any]:
    r"""Basic degree statistics of an undirected simple graph.

    The graph is binarised and symmetrised, so ``degree`` is the number of
    distinct neighbours (self-loops ignored).

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency.

    Returns
    -------
    dict
        Keys:

        * ``num_nodes`` (int),
        * ``num_edges`` (int, undirected),
        * ``mean_degree`` (float),
        * ``max_degree`` (int),
        * ``min_degree`` (int),
        * ``degrees`` (list[int], per-node degree),
        * ``histogram`` (dict[int, int], degree -> count of nodes).

    Raises
    ------
    ValueError
        If ``adj`` is not square.
    """
    a = _binary_undirected(adj)
    n = a.shape[0]
    deg = a.sum(axis=1).astype(int)
    counts = np.bincount(deg, minlength=1) if n > 0 else np.zeros(1, dtype=int)
    histogram = {int(d): int(c) for d, c in enumerate(counts) if c > 0}
    return {
        "num_nodes": int(n),
        "num_edges": int(deg.sum() // 2),
        "mean_degree": float(deg.mean()) if n > 0 else 0.0,
        "max_degree": int(deg.max()) if n > 0 else 0,
        "min_degree": int(deg.min()) if n > 0 else 0,
        "degrees": deg.tolist(),
        "histogram": histogram,
    }
