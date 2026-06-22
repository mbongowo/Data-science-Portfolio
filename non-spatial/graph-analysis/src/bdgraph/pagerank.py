r"""PageRank by power iteration on a dense adjacency matrix.

This is the **pure-numpy reference layer**: no third-party dependency beyond
numpy, always importable, and the basis of the known-answer tests. It is meant
for small graphs, teaching, and validation. The same algorithm runs at scale on
Spark in :mod:`bdgraph.graphframes_pipeline`.

PageRank is the stationary distribution of a random surfer who, with
probability ``damping``, follows an out-edge chosen uniformly at random and,
with probability ``1 - damping``, teleports to a node chosen uniformly. The
iteration is

.. math::

    p^{(t+1)} = (1 - d)\,\frac{1}{n}\mathbf{1}
                + d\, \big(M p^{(t)} + \tfrac{1}{n}\mathbf{1}\, s^\top p^{(t)}\big)

where ``M`` is the column-stochastic transition matrix (column ``j`` is the
out-distribution of node ``j``), ``d`` is the damping factor, and ``s`` flags
**dangling nodes** (no out-edges). A dangling node's mass is redistributed
uniformly across all nodes, which keeps ``p`` a probability vector summing to 1.

Interpretation notes (do not skip these):

* PageRank measures the steady-state visit frequency of a *specific* random
  walk. It is not "importance" in any absolute sense, and changing the damping
  factor or the teleport set changes the ranking.
* The result is conditional on the edge set and its direction. Reversing edges,
  or treating a directed graph as undirected, generally changes the ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def pagerank(
    adj: ArrayLike,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> NDArray[np.float64]:
    r"""Compute PageRank on a dense adjacency matrix via power iteration.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency matrix. ``adj[i, j]`` is the weight of the
        edge **from** ``i`` **to** ``j``. The diagonal is ignored (self-loops
        do not contribute). Weights may be non-negative reals, not just 0/1.
    damping:
        Probability of following an out-edge rather than teleporting. The
        classic value is ``0.85``. Must satisfy ``0 <= damping <= 1``.
    max_iter:
        Maximum number of power-iteration steps.
    tol:
        Convergence threshold on the L1 distance between successive iterates.

    Returns
    -------
    numpy.ndarray
        Length-``n`` PageRank vector. It is non-negative and sums to 1.

    Raises
    ------
    ValueError
        If ``adj`` is not square, ``n < 1``, ``damping`` is outside ``[0, 1]``,
        or any edge weight is negative.

    Notes
    -----
    Dangling nodes (rows that sum to zero) have their probability mass spread
    uniformly over all nodes each iteration, so the vector stays normalised.
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

    out_strength = a.sum(axis=1)
    dangling = out_strength == 0.0

    # Column-stochastic transition matrix M: column j is node j's out-distribution.
    # Start from the row-normalised out-links of each node, then transpose.
    row_norm = np.zeros_like(a)
    live = ~dangling
    row_norm[live] = a[live] / out_strength[live][:, None]
    m = row_norm.T  # m[i, j] = prob of moving to i given currently at j

    teleport = np.full(n, 1.0 / n)
    p = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        # Mass sitting on dangling nodes is redistributed uniformly.
        dangling_mass = float(p[dangling].sum())
        p_next = (1.0 - damping) * teleport + damping * (
            m @ p + dangling_mass * teleport
        )
        # Guard against tiny floating-point drift in the total mass.
        p_next /= p_next.sum()
        if np.abs(p_next - p).sum() < tol:
            p = p_next
            break
        p = p_next

    return p
