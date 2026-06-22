r"""Community detection by label propagation.

Part of the **pure-numpy reference layer**: numpy + stdlib only, always
importable, covered by hand-derived known-answer tests. The Spark equivalent
(label propagation, LPA) lives in :mod:`bdgraph.graphframes_pipeline`.

Label propagation (Raghavan, Albert & Kumara 2007) is a near-linear community
heuristic: every node starts with its own label; each round, a node adopts the
label held by the largest total edge weight among its neighbours; ties are
broken **deterministically** by choosing the lowest label. With a fixed seed
controlling the node visit order, the result is reproducible run to run.

Interpretation notes (do not skip these):

* Label propagation has no objective function it provably optimises, and like
  modularity methods it has a **resolution limit**: it can merge small genuine
  communities or, on near-symmetric graphs, oscillate. The deterministic
  tie-break here removes the randomness but not the heuristic's bias.
* "Community" here means densely interconnected nodes, not groups with any
  external meaning. The labels are arbitrary integers, not ranks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def label_propagation(
    adj: ArrayLike, max_iter: int = 10, seed: int = 0
) -> NDArray[np.int_]:
    r"""Detect communities by deterministic synchronous-ish label propagation.

    The graph is treated as **undirected**: the adjacency is symmetrised before
    propagation. Each round, every node (visited in a seeded random order)
    adopts the neighbour label carrying the greatest summed edge weight; ties
    are broken by the **lowest** label so the outcome is deterministic. The
    sweep repeats until labels stop changing or ``max_iter`` is reached.

    Parameters
    ----------
    adj:
        Dense ``n`` x ``n`` adjacency matrix; ``adj[i, j]`` is the weight of the
        edge between ``i`` and ``j``. The diagonal is ignored.
    max_iter:
        Maximum number of propagation sweeps.
    seed:
        Seed for the node visit order. The tie-break is independent of the seed,
        so the final labelling is reproducible.

    Returns
    -------
    numpy.ndarray
        Length-``n`` integer label per node. Nodes with the same label are in
        the same community. Labels are remapped to the smallest member id of
        each community so the labelling is canonical.

    Raises
    ------
    ValueError
        If ``adj`` is not square or has negative weights.
    """
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be a square 2-D matrix, got {a.shape}.")
    if (a < 0).any():
        raise ValueError("Adjacency weights must be non-negative.")

    n = a.shape[0]
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    a = np.maximum(a, a.T)  # symmetrise: undirected propagation

    labels = np.arange(n)
    rng = np.random.default_rng(seed)

    for _ in range(max_iter):
        changed = False
        order = rng.permutation(n)
        for i in order:
            neigh = np.flatnonzero(a[i] > 0)
            if neigh.size == 0:
                continue
            # Sum edge weight per candidate label among this node's neighbours.
            weight_by_label: dict[int, float] = {}
            for j in neigh:
                lab = int(labels[j])
                weight_by_label[lab] = weight_by_label.get(lab, 0.0) + float(a[i, j])
            # Pick the heaviest label; break ties by the lowest label id.
            best = min(weight_by_label, key=lambda lab: (-weight_by_label[lab], lab))
            if best != labels[i]:
                labels[i] = best
                changed = True
        if not changed:
            break

    return _canonical_labels(labels)


def _canonical_labels(labels: NDArray[np.int_]) -> NDArray[np.int_]:
    """Remap labels so each community is named by its smallest member id."""
    out = labels.copy()
    smallest: dict[int, int] = {}
    for idx, lab in enumerate(labels):
        lab = int(lab)
        if lab not in smallest:
            smallest[lab] = idx
    for idx in range(len(labels)):
        out[idx] = smallest[int(labels[idx])]
    return out
