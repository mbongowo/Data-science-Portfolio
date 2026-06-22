"""Known-answer tests for deterministic label propagation.

Hand-reasoned on two disjoint cliques: no edge crosses between them, so labels
can never propagate across the gap and the algorithm must return exactly two
communities. The lowest-label tie-break plus a fixed seed make the labelling
reproducible run to run. numpy only.

Graph: K3 on {0,1,2} and K3 on {3,4,5}, no cross edges.
"""

from __future__ import annotations

import numpy as np

from bdgraph.community import label_propagation


def _two_cliques() -> np.ndarray:
    a = np.zeros((6, 6))
    for i, j in [(0, 1), (0, 2), (1, 2)]:  # first K3
        a[i, j] = a[j, i] = 1.0
    for i, j in [(3, 4), (3, 5), (4, 5)]:  # second K3
        a[i, j] = a[j, i] = 1.0
    return a


def test_two_cliques_give_two_communities() -> None:
    labels = label_propagation(_two_cliques(), max_iter=10, seed=0)
    assert len(np.unique(labels)) == 2
    # Members of each clique share a label; the two labels differ.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_deterministic_across_runs() -> None:
    a = _two_cliques()
    first = label_propagation(a, max_iter=10, seed=0)
    second = label_propagation(a, max_iter=10, seed=0)
    assert np.array_equal(first, second)


def test_canonical_label_is_smallest_member() -> None:
    """Communities are named by their smallest member id."""
    labels = label_propagation(_two_cliques(), max_iter=10, seed=0)
    assert labels[0] == 0
    assert labels[3] == 3


def test_disconnected_nodes_keep_own_label() -> None:
    """Isolated nodes form singleton communities."""
    a = np.zeros((3, 3))
    a[0, 1] = a[1, 0] = 1.0  # node 2 is isolated
    labels = label_propagation(a, max_iter=10, seed=0)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert len(np.unique(labels)) == 2


def test_invalid_inputs_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        label_propagation(np.zeros((2, 3)))  # not square
    with pytest.raises(ValueError):
        label_propagation(np.array([[0.0, -1.0], [-1.0, 0.0]]))  # negative weight
