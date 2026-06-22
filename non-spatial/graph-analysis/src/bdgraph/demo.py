r"""One-command demo: drive the pure-numpy core end-to-end on a seeded graph.

This module synthesises a small **stochastic block model (SBM)** graph with a
known, *planted* community structure and runs the real reference-layer
algorithms on it: PageRank, connected components, deterministic label
propagation, and triangle counting. Everything is deterministic in ``seed`` and
runs in well under a second, with **no** third-party dependency beyond numpy /
pandas / stdlib -- so it runs anywhere, including CI.

The point is honesty: the numbers reported in the README come from *this* code,
not from a hand-typed guess. The same algorithms (`bdgraph.graphframes_pipeline`)
run at scale on a real SNAP network; this demo exercises the identical numeric
core on a graph small enough to verify by eye.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bdgraph.centrality import betweenness_centrality
from bdgraph.community import label_propagation
from bdgraph.components import num_components
from bdgraph.pagerank import pagerank
from bdgraph.structure import degree_stats, k_core_decomposition, modularity
from bdgraph.triangles import per_node_triangles, triangle_count

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray


def synthesize_sbm(
    *,
    sizes: tuple[int, ...] = (12, 10, 8),
    p_in: float = 0.6,
    p_out: float = 0.02,
    seed: int = 0,
) -> NDArray[np.float64]:
    r"""Build a symmetric 0/1 adjacency for a planted stochastic block model.

    Nodes are partitioned into blocks of the given ``sizes``. A pair of nodes in
    the **same** block is connected with probability ``p_in`` (dense
    communities); a pair in **different** blocks with probability ``p_out``
    (sparse inter-community links). The result is a simple undirected graph
    (symmetric, zero diagonal). All randomness flows from
    ``numpy.random.default_rng(seed)`` so the graph is reproducible.

    Parameters
    ----------
    sizes:
        Number of nodes in each planted community.
    p_in, p_out:
        Edge probabilities within and between communities.
    seed:
        Seed for the dense Bernoulli draws.

    Returns
    -------
    numpy.ndarray
        Dense ``n`` x ``n`` symmetric 0/1 adjacency, ``n = sum(sizes)``.
    """
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))

    # Per-node block id, e.g. [0,0,...,1,1,...,2,2,...].
    block = np.concatenate([np.full(s, b) for b, s in enumerate(sizes)])
    same_block = block[:, None] == block[None, :]
    prob = np.where(same_block, p_in, p_out)

    draws = rng.random((n, n))
    a = (draws < prob).astype(float)
    a = np.triu(a, k=1)  # keep upper triangle only ...
    a = a + a.T  # ... then mirror for a symmetric simple graph
    np.fill_diagonal(a, 0.0)
    return a


def _planted_labels(sizes: tuple[int, ...]) -> NDArray[np.int_]:
    """Ground-truth community id per node (canonical: smallest member id)."""
    return np.concatenate(
        [np.full(s, int(np.sum(sizes[:b]))) for b, s in enumerate(sizes)]
    )


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    r"""Run the whole pure-numpy core on a seeded SBM graph and write artifacts.

    Synthesises a small planted-community graph (see :func:`synthesize_sbm`),
    runs the four reference algorithms, writes ``pagerank_top.csv``,
    ``communities.csv`` and ``summary.json`` into ``out_dir``, and returns a
    summary dict of the headline metrics.

    Parameters
    ----------
    seed:
        Seed controlling both the synthetic graph and the label-propagation
        visit order. The whole pipeline is deterministic in ``seed``.
    out_dir:
        Directory for the CSV / JSON artifacts. Created if absent.

    Returns
    -------
    dict
        Keys: ``num_nodes``, ``num_edges``, ``num_components``,
        ``num_communities_found``, ``num_planted_communities``,
        ``global_triangles``, ``avg_clustering``, ``max_core_number``,
        ``modularity_found``, ``modularity_planted``, ``top_betweenness_node``,
        ``mean_degree``, ``max_degree`` and ``top_pagerank`` (the top-5
        ``(node_id, score)`` pairs).
    """
    sizes = (12, 10, 8)
    a = synthesize_sbm(sizes=sizes, seed=seed)
    n = a.shape[0]

    # Edge list (undirected, each pair once) from the upper triangle.
    iu, ju = np.where(np.triu(a, k=1) > 0)
    edges = [(int(i), int(j)) for i, j in zip(iu, ju, strict=True)]
    num_edges = len(edges)

    # --- real core ---------------------------------------------------------
    pr = pagerank(a, damping=0.85)
    order = np.argsort(pr)[::-1]
    top5 = [(int(i), float(pr[i])) for i in order[:5]]

    n_components = num_components(n, edges)

    labels = label_propagation(a, max_iter=20, seed=seed)
    n_found = int(np.unique(labels).size)
    planted = _planted_labels(sizes)
    n_planted = int(np.unique(planted).size)

    global_tri = triangle_count(a)
    per_node = per_node_triangles(a)

    # --- structural diagnostics on the same graph --------------------------
    core = k_core_decomposition(a)
    max_core = int(core.max())
    bc = betweenness_centrality(a, normalized=True)
    top_between = int(np.argmax(bc))
    # Modularity of the *detected* partition vs. a single trivial partition.
    q_found = modularity(a, labels)
    q_planted = modularity(a, planted)
    deg_stats = degree_stats(a)

    # Average local clustering coefficient: for each node with degree >= 2,
    # (triangles through it) / (possible pairs of its neighbours).
    deg = (a > 0).sum(axis=1).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        pairs = deg * (deg - 1) / 2.0
        local_cc = np.where(pairs > 0, per_node / pairs, 0.0)
    avg_clustering = float(local_cc.mean())

    # --- artifacts ---------------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "pagerank_top.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "node", "pagerank"])
        for rank, i in enumerate(order, start=1):
            w.writerow([rank, int(i), f"{float(pr[i]):.6f}"])

    with open(out / "communities.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["node", "found_community", "planted_community"])
        for node in range(n):
            w.writerow([node, int(labels[node]), int(planted[node])])

    summary: dict[str, Any] = {
        "seed": seed,
        "num_nodes": n,
        "num_edges": num_edges,
        "num_components": n_components,
        "num_communities_found": n_found,
        "num_planted_communities": n_planted,
        "global_triangles": global_tri,
        "avg_clustering": round(avg_clustering, 6),
        "max_core_number": max_core,
        "modularity_found": round(q_found, 6),
        "modularity_planted": round(q_planted, 6),
        "top_betweenness_node": top_between,
        "mean_degree": round(deg_stats["mean_degree"], 6),
        "max_degree": deg_stats["max_degree"],
        "top_pagerank": [
            {"node": node, "pagerank": round(score, 6)} for node, score in top5
        ],
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary["top_pagerank"] = top5  # richer tuples for the return value
    return summary
