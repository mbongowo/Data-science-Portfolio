"""Known-answer tests for the pure-numpy PageRank reference.

These use *hand-derived* stationary distributions on tiny graphs, so a green
test proves the power iteration converges to the right vector, not merely that
it runs. No third-party dependency beyond numpy.

Worked examples:

1. Two nodes, symmetric: 0 <-> 1. Each node's only out-edge points at the
   other, so the chain is doubly symmetric and the unique stationary vector is
   [1/2, 1/2] for any damping.

2. Directed 3-cycle: 0 -> 1 -> 2 -> 0. Every node has out-degree 1 and in-degree
   1 and the graph is vertex-transitive, so PageRank is uniform [1/3, 1/3, 1/3]
   for any damping.

3. A dangling node (no out-edges) must have its mass redistributed so the
   vector still sums to exactly 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from bdgraph.pagerank import pagerank

# 0 <-> 1
TWO_NODE_SYMMETRIC = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
)

# directed cycle 0 -> 1 -> 2 -> 0
THREE_CYCLE = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)


def test_two_node_symmetric_is_half_half() -> None:
    """A symmetric 2-node graph splits rank evenly (hand-derived)."""
    pr = pagerank(TWO_NODE_SYMMETRIC)
    assert pr == pytest.approx([0.5, 0.5], abs=1e-9)


def test_directed_cycle_is_uniform() -> None:
    """A directed 3-cycle is vertex-transitive => uniform PageRank."""
    pr = pagerank(THREE_CYCLE)
    assert pr == pytest.approx([1.0 / 3.0] * 3, abs=1e-9)


def test_pagerank_sums_to_one() -> None:
    """The PageRank vector is a probability distribution."""
    pr = pagerank(THREE_CYCLE)
    assert float(pr.sum()) == pytest.approx(1.0, abs=1e-12)
    assert (pr >= 0).all()


def test_uniform_independent_of_damping() -> None:
    """Vertex-transitivity forces uniformity at any damping factor."""
    for d in (0.0, 0.5, 0.85, 1.0):
        pr = pagerank(THREE_CYCLE, damping=d)
        assert pr == pytest.approx([1.0 / 3.0] * 3, abs=1e-9)


def test_dangling_node_handled() -> None:
    """A node with no out-edges must not leak probability mass.

    Graph: 0 -> 1, and node 1 dangles (no out-edges). The vector must still sum
    to 1, and node 1 (the only sink) must carry the larger share.
    """
    adj = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],  # node 1 is dangling
        ]
    )
    pr = pagerank(adj)
    assert float(pr.sum()) == pytest.approx(1.0, abs=1e-12)
    assert (pr >= 0).all()
    assert pr[1] > pr[0]


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        pagerank(np.zeros((2, 3)))  # not square
    with pytest.raises(ValueError):
        pagerank(TWO_NODE_SYMMETRIC, damping=1.5)  # damping out of range
    with pytest.raises(ValueError):
        pagerank(np.array([[0.0, -1.0], [0.0, 0.0]]))  # negative weight
