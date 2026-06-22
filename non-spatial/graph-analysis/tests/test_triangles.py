"""Known-answer tests for triangle counting via trace(A^3)/6.

Hand-derived on tiny graphs. numpy only.

* A single triangle (K3) has exactly 1 triangle; each of its 3 nodes is in 1.
* The complete graph K4 has C(4, 3) = 4 triangles.
* A path 0--1--2--3 has no triangle.
"""

from __future__ import annotations

import numpy as np

from bdgraph.triangles import per_node_triangles, triangle_count

# K3 on nodes 0,1,2
TRIANGLE = np.array(
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ],
    dtype=float,
)

# K4 on nodes 0,1,2,3
K4 = np.array(
    [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ],
    dtype=float,
)

# path 0--1--2--3
PATH_4 = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=float,
)


def test_single_triangle_is_one() -> None:
    assert triangle_count(TRIANGLE) == 1


def test_k4_has_four_triangles() -> None:
    assert triangle_count(K4) == 4


def test_path_has_no_triangle() -> None:
    assert triangle_count(PATH_4) == 0


def test_per_node_triangle_all_one() -> None:
    counts = per_node_triangles(TRIANGLE)
    assert counts.tolist() == [1, 1, 1]


def test_per_node_sums_to_three_times_total() -> None:
    """Each triangle touches 3 nodes, so the per-node counts sum to 3T."""
    assert int(per_node_triangles(K4).sum()) == 3 * triangle_count(K4)


def test_direction_and_diagonal_ignored() -> None:
    """A directed/self-looped variant of K3 still counts one triangle."""
    a = np.array(
        [
            [5, 1, 0],  # self-loop and a missing reverse edge
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    assert triangle_count(a) == 1
