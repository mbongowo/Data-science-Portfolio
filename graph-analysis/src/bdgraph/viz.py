"""Small-subgraph visualisation (networkx + matplotlib).

A 69M-edge graph cannot be drawn; any picture is of a **sample**. This wrapper
extracts a representative subgraph and lays it out for a sanity-check figure,
not for analysis. The heavy imports (``networkx``, ``matplotlib``) are lazy and
happen inside the functions, so the core package and the test suite never import
them. This module is not imported by :mod:`bdgraph` or by the tests.

A drawn sample can mislead: a breadth-first or ego sample over-represents
high-degree hubs and the neighbourhoods near the seed, so the picture is not an
unbiased view of the whole graph. Treat it as illustration only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx


def sample_subgraph(
    edges: Iterable[tuple[int, int]],
    *,
    seed_node: int | None = None,
    max_nodes: int = 200,
    seed: int = 0,
) -> nx.Graph:
    """Return a connected subgraph of at most ``max_nodes`` nodes.

    Grows a breadth-first ball from ``seed_node`` (or a randomly chosen node)
    until the node budget is hit. The result is a ``networkx.Graph`` suitable
    for :func:`draw`.
    """
    import random
    from collections import deque

    import networkx as nx

    g = nx.Graph()
    g.add_edges_from(edges)
    if g.number_of_nodes() == 0:
        return g

    rng = random.Random(seed)
    start = seed_node if seed_node is not None else rng.choice(list(g.nodes))

    visited: set[int] = {start}
    queue: deque[int] = deque([start])
    while queue and len(visited) < max_nodes:
        node = queue.popleft()
        for nbr in g.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
                if len(visited) >= max_nodes:
                    break
    return g.subgraph(visited).copy()


def draw(
    subgraph: nx.Graph,
    *,
    out_path: str | None = None,
    node_attr: dict[int, Any] | None = None,
    seed: int = 0,
) -> Any:
    """Draw a subgraph with a spring layout; return the matplotlib axis.

    If ``node_attr`` is given (e.g. PageRank scores or community labels) it is
    used to size or colour nodes. With ``out_path`` the figure is saved.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(subgraph, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = 50.0
    colors: Any = "#4c72b0"
    if node_attr is not None:
        vals = [float(node_attr.get(n, 0.0)) for n in subgraph.nodes]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        span = (hi - lo) or 1.0
        sizes = [50.0 + 450.0 * (v - lo) / span for v in vals]
        colors = vals

    nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax, node_size=sizes, node_color=colors, cmap="viridis"
    )
    ax.set_axis_off()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return ax
