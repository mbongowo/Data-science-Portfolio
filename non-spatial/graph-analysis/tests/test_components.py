"""Known-answer tests for union-find connected components.

Hand-derived on a tiny graph with two clusters and a guaranteed-canonical
labelling (smallest member id per component). Stdlib only.

Graph (n = 5):
    cluster A: 0 -- 1 -- 2      (edges (0,1), (1,2))
    cluster B: 3 -- 4           (edge (3,4))
Expected labels: [0, 0, 0, 3, 3]; two components.
"""

from __future__ import annotations

from bdgraph.components import connected_components, num_components

TWO_CLUSTERS = [(0, 1), (1, 2), (3, 4)]


def test_two_clusters_labels() -> None:
    labels = connected_components(5, TWO_CLUSTERS)
    assert labels == [0, 0, 0, 3, 3]


def test_two_clusters_count() -> None:
    assert num_components(5, TWO_CLUSTERS) == 2


def test_isolated_nodes_are_their_own_component() -> None:
    # n = 4, single edge (0, 1); nodes 2 and 3 are isolated.
    labels = connected_components(4, [(0, 1)])
    assert labels == [0, 0, 2, 3]
    assert num_components(4, [(0, 1)]) == 3


def test_direction_is_ignored() -> None:
    forward = connected_components(3, [(0, 1), (1, 2)])
    backward = connected_components(3, [(1, 0), (2, 1)])
    assert forward == backward == [0, 0, 0]


def test_fully_connected_is_one_component() -> None:
    assert num_components(4, [(0, 1), (1, 2), (2, 3)]) == 1


def test_no_edges_all_singletons() -> None:
    assert connected_components(3, []) == [0, 1, 2]
    assert num_components(3, []) == 3


def test_out_of_range_endpoint_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        connected_components(2, [(0, 5)])
