r"""Weighted / personalized PageRank and Brandes betweenness centrality.

Part of the **pure-numpy reference layer**: numpy + stdlib only, always
importable, and covered by hand-derived known-answer tests. These are the
centrality measures a reviewer expects alongside plain PageRank, kept small and
exact so the results can be checked by eye.

Two families live here:

* **Weighted / personalized PageRank** (:func:`weighted_pagerank`,
  :func:`personalized_pagerank`). The plain :func:`bdgraph.pagerank.pagerank`
  already honours edge weights through the out-strength normalisation; these
  wrappers expose that explicitly and add a **personalization / restart**
  vector, so teleport mass lands on a chosen distribution rather than uniformly.
  With uniform weights and a uniform restart they reduce exactly to plain
  PageRank.

* **Betweenness centrality** (:func:`betweenness_centrality`) by **Brandes'
  algorithm**. For each source we run a BFS, count shortest paths, and
  back-propagate dependencies; summing over sources gives the exact betweenness
  in :math:`O(nm)` time. This is fine for the small graphs the reference layer
  targets, not for SNAP-scale networks.

Interpretation notes (do not skip these):

* Personalized PageRank answers "important *relative to this restart set*", not
  "important in general". Move the restart mass and the ranking moves with it.
* Betweenness measures how often a node lies on shortest paths. It is sensitive
  to a handful of bridges and, like every centrality, is conditional on the edge
  set; it does not measure importance in any absolute sense.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from bdgraph.pagerank import pagerank

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def weighted_pagerank(
    adj: ArrayLike,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> NDArray[np.float64]:
    r"""PageRank that honours edge weights (alias-clear wrapper).

    Identical numerics to :func:`bdgraph.pagerank.pagerank`: the transition
    matrix is built by normalising each node's out-edges by its **out-strength**
    (the row sum of weights), so a heavier edge carries proportionally more of
    the surfer's mass. This wrapper exists to make the weighted semantics
    explicit in the public API and to document that uniform 0/1 weights reduce
    to the classic unweighted PageRank.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` weighted adjacency. ``adj[i, j]`` is the weight of
        the edge from ``i`` to ``j``; weights must be non-negative. The diagonal
        (self-loops) is ignored.
    damping, max_iter, tol:
        As in :func:`bdgraph.pagerank.pagerank`.

    Returns
    -------
    numpy.ndarray
        Length-``n`` PageRank vector; non-negative, sums to 1.
    """
    return pagerank(adj, damping=damping, max_iter=max_iter, tol=tol)


def personalized_pagerank(
    adj: ArrayLike,
    restart: ArrayLike,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> NDArray[np.float64]:
    r"""Personalized (restart-biased) PageRank on a weighted adjacency.

    Same random surfer as plain PageRank, except that when the surfer teleports
    (probability ``1 - damping``) it lands on a node drawn from the
    **restart distribution** rather than uniformly. Dangling-node mass is also
    redistributed along the restart vector, which keeps the result a probability
    vector. The iteration is

    .. math::

        p^{(t+1)} = (1 - d)\,r + d\,\big(M p^{(t)} + (s^\top p^{(t)})\,r\big)

    where ``M`` is the column-stochastic weighted transition matrix, ``r`` is the
    normalised restart vector, ``d`` the damping, and ``s`` flags dangling nodes.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` weighted adjacency (non-negative; diagonal ignored).
    restart:
        Length-``n`` non-negative restart / personalization weights. They are
        normalised to sum to 1 internally and must not be all zero. A uniform
        restart reproduces plain PageRank exactly.
    damping, max_iter, tol:
        As in :func:`bdgraph.pagerank.pagerank`.

    Returns
    -------
    numpy.ndarray
        Length-``n`` personalized PageRank vector; non-negative, sums to 1.

    Raises
    ------
    ValueError
        If ``adj`` is not square, ``damping`` is outside ``[0, 1]``, any weight
        is negative, the restart length is wrong, or the restart sums to zero.
    """
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be a square 2-D matrix, got {a.shape}.")
    n = a.shape[0]
    if n < 1:
        raise ValueError("PageRank requires at least one node.")
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be in [0, 1].")

    a = a.copy()
    np.fill_diagonal(a, 0.0)
    if (a < 0).any():
        raise ValueError("Adjacency weights must be non-negative.")

    r = np.asarray(restart, dtype=float).ravel()
    if r.shape[0] != n:
        raise ValueError(f"restart must have length {n}, got {r.shape[0]}.")
    if (r < 0).any():
        raise ValueError("restart weights must be non-negative.")
    total = float(r.sum())
    if total <= 0.0:
        raise ValueError("restart must not sum to zero.")
    r = r / total

    out_strength = a.sum(axis=1)
    dangling = out_strength == 0.0

    row_norm = np.zeros_like(a)
    live = ~dangling
    row_norm[live] = a[live] / out_strength[live][:, None]
    m = row_norm.T  # column-stochastic: m[i, j] = P(move to i | at j)

    p = r.copy()
    for _ in range(max_iter):
        dangling_mass = float(p[dangling].sum())
        p_next = (1.0 - damping) * r + damping * (m @ p + dangling_mass * r)
        p_next /= p_next.sum()
        if np.abs(p_next - p).sum() < tol:
            p = p_next
            break
        p = p_next

    return p


def betweenness_centrality(
    adj: ArrayLike, normalized: bool = False
) -> NDArray[np.float64]:
    r"""Exact shortest-path betweenness centrality (Brandes' algorithm).

    The betweenness of node ``v`` is the sum, over all ordered source/target
    pairs ``(s, t)`` with ``s != v != t``, of the fraction of shortest
    ``s``-``t`` paths that pass through ``v``:

    .. math::

        C_B(v) = \sum_{s \neq v \neq t}
                 \frac{\sigma_{st}(v)}{\sigma_{st}}.

    The graph is treated as **undirected and unweighted** (edges binarised and
    symmetrised); shortest paths are found by BFS. Brandes' back-propagation
    accumulates each source's dependency in one pass, giving exact betweenness
    in :math:`O(nm)`.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency. Binarised (any nonzero is an edge),
        symmetrised, diagonal cleared.
    normalized:
        If ``True``, divide by ``(n-1)(n-2)/2`` (the number of **unordered**
        pairs not involving ``v``) so scores lie in ``[0, 1]``. Default
        ``False`` returns the raw unordered-pair count; for an undirected star
        this gives ``(n-1)(n-2)/2`` at the centre.

    Returns
    -------
    numpy.ndarray
        Length-``n`` betweenness scores.

    Raises
    ------
    ValueError
        If ``adj`` is not square.

    Notes
    -----
    The undirected convention counts each unordered pair ``{s, t}`` once (the
    raw Brandes accumulation, summed over both endpoints, is halved). The raw
    score for an undirected star centre is then ``(n-1)(n-2)/2``; path-graph
    endpoints score 0, never lying between any other pair.
    """
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be a square 2-D matrix, got {a.shape}.")
    n = a.shape[0]
    a = (a != 0).astype(float)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)

    # Adjacency lists for BFS.
    neighbours = [np.flatnonzero(a[i] > 0).tolist() for i in range(n)]
    bc = np.zeros(n, dtype=float)

    for s in range(n):
        stack: list[int] = []
        preds: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)  # number of shortest paths from s
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=int)
        dist[s] = 0
        queue: deque[int] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in neighbours[v]:
                if dist[w] < 0:  # first time we reach w
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:  # shortest-path edge v -> w
                    sigma[w] += sigma[v]
                    preds[w].append(v)

        # Back-propagate dependencies (Brandes accumulation).
        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # Each unordered pair {s, t} was accumulated from both s and t, so halve to
    # the standard undirected convention (one count per unordered pair).
    bc /= 2.0

    if normalized:
        if n > 2:
            # (n-1)(n-2)/2 unordered pairs not involving v.
            bc /= (n - 1) * (n - 2) / 2.0
        else:
            bc[:] = 0.0
    return bc
