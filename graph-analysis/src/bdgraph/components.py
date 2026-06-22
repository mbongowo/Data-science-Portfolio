r"""Connected components by union-find (disjoint-set).

Part of the **pure-numpy reference layer**: stdlib + numpy only, always
importable, and covered by hand-derived known-answer tests. The Spark
equivalent (`connectedComponents`) lives in
:mod:`bdgraph.graphframes_pipeline`.

The graph is treated as **undirected** for connectivity: an edge ``(u, v)``
joins ``u`` and ``v`` regardless of stored direction. Two nodes are in the same
component if a path of edges connects them.

Interpretation note: connected components describe reachability under the
undirected edge set only. On a directed graph this is *weak* connectivity, not
strong connectivity; a pair of nodes in the same weak component need not be able
to reach each other respecting edge direction.
"""

from __future__ import annotations

from collections.abc import Iterable


def _find(parent: list[int], x: int) -> int:
    """Find the root of ``x`` with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression: point every node on the path straight at the root.
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    """Label the connected components of an undirected graph.

    Parameters
    ----------
    n:
        Number of nodes. Nodes are ``0 .. n - 1``.
    edges:
        Iterable of ``(u, v)`` integer pairs. Direction is ignored; ``(u, v)``
        and ``(v, u)`` have the same effect. Self-loops are harmless.

    Returns
    -------
    list[int]
        Length-``n`` list of component labels. Two nodes share a label iff they
        are connected. Labels are the smallest node id in each component, so the
        labelling is canonical and deterministic.

    Raises
    ------
    ValueError
        If ``n < 0`` or any endpoint is outside ``0 .. n - 1``.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")

    parent = list(range(n))
    rank = [0] * n

    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u}, {v}) has an endpoint outside 0..{n - 1}.")
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            continue
        # Union by rank to keep the trees shallow.
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1

    # Canonical label = smallest member of the component.
    label = list(range(n))
    for x in range(n):
        root = _find(parent, x)
        if x < label[root]:
            label[root] = x
    return [label[_find(parent, x)] for x in range(n)]


def num_components(n: int, edges: Iterable[tuple[int, int]]) -> int:
    """Return the number of connected components.

    A convenience wrapper over :func:`connected_components` that counts the
    distinct labels.
    """
    return len(set(connected_components(n, edges)))
