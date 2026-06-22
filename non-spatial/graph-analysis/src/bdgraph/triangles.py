r"""Triangle counting for undirected simple graphs.

Part of the **pure-numpy reference layer**: numpy only, always importable,
covered by hand-derived known-answer tests. The Spark equivalent
(`triangleCount`) lives in :mod:`bdgraph.graphframes_pipeline`.

For an undirected simple graph with binary adjacency ``A`` (symmetric, zero
diagonal), the number of closed triangles is

.. math::

    T = \frac{1}{6}\,\operatorname{tr}(A^3),

and the number of triangles **through node** ``i`` is

.. math::

    t_i = \tfrac{1}{2}\,(A^3)_{ii}.

The factor ``1/6`` removes the six orderings (3! permutations) of each
triangle's vertices counted by the trace; the per-node factor ``1/2`` removes
the two directions of traversal.

Interpretation note: triangle counts and the clustering they imply describe
local link density, not cause. A high global triangle count says the graph is
locally clustered; it says nothing about why edges formed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def _binary_undirected(adj: ArrayLike) -> NDArray[np.float64]:
    """Return a symmetric 0/1 copy of ``adj`` with a zero diagonal."""
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be a square 2-D matrix, got {a.shape}.")
    a = (a != 0).astype(float)  # binarise: simple graph, ignore multiplicities
    a = np.maximum(a, a.T)  # symmetrise: undirected
    np.fill_diagonal(a, 0.0)
    return a


def triangle_count(adj: ArrayLike) -> int:
    r"""Count closed triangles in an undirected simple graph.

    Computes :math:`\operatorname{tr}(A^3) / 6` on the binarised, symmetrised
    adjacency. The result is exact for integer-valued counts; it is rounded to
    the nearest integer to absorb floating-point error from the matrix products.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency matrix. It is binarised (any nonzero entry
        is an edge), symmetrised, and the diagonal is cleared.

    Returns
    -------
    int
        The number of distinct triangles (3-cliques).
    """
    a = _binary_undirected(adj)
    a3 = a @ a @ a
    return int(round(float(np.trace(a3)) / 6.0))


def per_node_triangles(adj: ArrayLike) -> NDArray[np.int_]:
    r"""Count triangles through each node.

    Returns :math:`\operatorname{diag}(A^3) / 2` on the binarised, symmetrised
    adjacency, rounded to integers. The sum of the per-node counts equals three
    times :func:`triangle_count`, since each triangle is incident to three
    nodes.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency matrix (binarised and symmetrised here).

    Returns
    -------
    numpy.ndarray
        Length-``n`` integer array; entry ``i`` is the number of triangles that
        include node ``i``.
    """
    a = _binary_undirected(adj)
    a3 = a @ a @ a
    diag = np.diag(a3) / 2.0
    return np.rint(diag).astype(int)
