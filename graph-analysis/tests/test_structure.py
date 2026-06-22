"""Known-answer tests for k-core, modularity and degree statistics.

Expected values are hand-derived on tiny graphs. numpy only.

k-core:
* Clique K_m: every node has core number m-1 (K4 -> 3).
* Path: every node has core number 1.
* Cycle: every node has core number 2.
* Isolated node: core number 0.

modularity:
* Two disjoint triangles, each its own community, give Q = 0.5 exactly.
* Everyone in one community gives Q = 0.
* A singleton-per-node partition gives a negative Q.

degree_stats:
* K4 -> every degree 3; histogram {3: 4}; 6 undirected edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from bdgraph.structure import degree_stats, k_core_decomposition, modularity


def _clique(m: int) -> np.ndarray:
    return np.ones((m, m)) - np.eye(m)


def _path(n: int) -> np.ndarray:
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _cycle(n: int) -> np.ndarray:
    a = _path(n)
    a[0, n - 1] = a[n - 1, 0] = 1.0
    return a


def _two_triangles() -> np.ndarray:
    """K3 on {0,1,2} and K3 on {3,4,5}, no cross edges."""
    a = np.zeros((6, 6))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a[i, j] = a[j, i] = 1.0
    for i, j in [(3, 4), (3, 5), (4, 5)]:
        a[i, j] = a[j, i] = 1.0
    return a


# --- k-core ----------------------------------------------------------------


def test_clique_core_is_size_minus_one() -> None:
    core = k_core_decomposition(_clique(4))
    assert core.tolist() == [3, 3, 3, 3]


def test_path_core_is_one() -> None:
    core = k_core_decomposition(_path(5))
    assert core.tolist() == [1, 1, 1, 1, 1]


def test_cycle_core_is_two() -> None:
    core = k_core_decomposition(_cycle(5))
    assert core.tolist() == [2, 2, 2, 2, 2]


def test_isolated_node_core_zero() -> None:
    a = _clique(3)
    a = np.pad(a, ((0, 1), (0, 1)))  # add isolated node 3
    core = k_core_decomposition(a)
    assert core[3] == 0
    assert core[:3].tolist() == [2, 2, 2]


def test_core_ignores_self_loops_and_direction() -> None:
    """Self-loops and asymmetry must not change the core numbers of K3."""
    a = np.array(
        [
            [5.0, 1.0, 0.0],  # self-loop, missing reverse edge
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    assert k_core_decomposition(a).tolist() == [2, 2, 2]


def test_core_single_node() -> None:
    assert k_core_decomposition(np.zeros((1, 1))).tolist() == [0]


def test_core_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        k_core_decomposition(np.zeros((2, 3)))


# --- modularity ------------------------------------------------------------


def test_two_triangles_modularity_is_half() -> None:
    """Two disjoint triangles, partitioned as themselves: Q = 0.5 (hand-derived).

    m = 6 edges. Each community has 3 internal edges and total degree 6, so
    sum_in/m - (sum_deg/2m)^2 per community = 3/6 - (6/12)^2 = 0.5 - 0.25 = 0.25;
    two communities sum to 0.5.
    """
    a = _two_triangles()
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert modularity(a, labels) == pytest.approx(0.5)


def test_single_community_modularity_zero() -> None:
    a = _two_triangles()
    labels = np.zeros(6, dtype=int)
    assert modularity(a, labels) == pytest.approx(0.0, abs=1e-12)


def test_singleton_partition_is_negative() -> None:
    """Every node its own community removes all within-edges => Q < 0."""
    a = _two_triangles()
    labels = np.arange(6)
    assert modularity(a, labels) < 0.0


def test_modularity_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        modularity(np.zeros((2, 3)), [0, 0])  # not square
    with pytest.raises(ValueError):
        modularity(_two_triangles(), [0, 0, 0])  # wrong label length


def test_modularity_edgeless_is_zero() -> None:
    assert modularity(np.zeros((3, 3)), [0, 1, 2]) == 0.0


# --- degree statistics -----------------------------------------------------


def test_degree_stats_k4() -> None:
    stats = degree_stats(_clique(4))
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 6
    assert stats["mean_degree"] == pytest.approx(3.0)
    assert stats["max_degree"] == 3
    assert stats["min_degree"] == 3
    assert stats["histogram"] == {3: 4}


def test_degree_stats_path() -> None:
    stats = degree_stats(_path(4))
    assert stats["degrees"] == [1, 2, 2, 1]
    assert stats["histogram"] == {1: 2, 2: 2}
    assert stats["num_edges"] == 3


def test_degree_stats_isolated_and_selfloop() -> None:
    """Self-loops are ignored; an isolated node has degree 0."""
    a = np.array(
        [
            [3.0, 1.0, 0.0],  # self-loop ignored
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # isolated
        ]
    )
    stats = degree_stats(a)
    assert stats["degrees"] == [1, 1, 0]
    assert stats["min_degree"] == 0
    assert stats["histogram"] == {0: 1, 1: 2}


def test_degree_stats_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        degree_stats(np.zeros((2, 3)))
