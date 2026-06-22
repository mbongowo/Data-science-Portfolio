"""Known-answer tests for weighted/personalized PageRank and betweenness.

All expected values are hand-derived on tiny graphs, so a green test proves the
algorithms compute the right thing, not merely that they run. numpy only.

PageRank:

* Uniform weights => weighted PageRank equals plain PageRank (no privilege).
* A heavier edge into one node draws more of the surfer's mass onto it.
* A uniform restart reproduces plain PageRank; a one-hot restart concentrates
  mass near the restart node.

Betweenness (Brandes, undirected, raw pair-count form):

* Star centre: the hub lies on every shortest path between the (n-1) leaves, so
  its raw betweenness is the number of ordered leaf pairs, (n-1)(n-2)/2 = 6 for
  a 5-node star (4 leaves).  Leaves score 0.
* Path endpoints lie between no other pair => 0; the middle node of a 3-path
  lies between the two ends (both orders) => 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from bdgraph.centrality import (
    betweenness_centrality,
    personalized_pagerank,
    weighted_pagerank,
)
from bdgraph.pagerank import pagerank

# directed 3-cycle 0 -> 1 -> 2 -> 0 (vertex-transitive)
THREE_CYCLE = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)


def _star(n: int) -> np.ndarray:
    """Undirected star: node 0 is the hub, nodes 1..n-1 are leaves."""
    a = np.zeros((n, n))
    for leaf in range(1, n):
        a[0, leaf] = a[leaf, 0] = 1.0
    return a


def _path(n: int) -> np.ndarray:
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


# --- weighted PageRank -----------------------------------------------------


def test_weighted_equals_plain_when_uniform() -> None:
    """Uniform 0/1 weights => weighted PageRank == plain PageRank."""
    wp = weighted_pagerank(THREE_CYCLE, damping=0.85)
    pp = pagerank(THREE_CYCLE, damping=0.85)
    assert wp == pytest.approx(pp, abs=1e-12)


def test_weighted_sums_to_one_nonnegative() -> None:
    a = np.array([[0.0, 3.0, 1.0], [1.0, 0.0, 1.0], [2.0, 2.0, 0.0]])
    wp = weighted_pagerank(a)
    assert float(wp.sum()) == pytest.approx(1.0, abs=1e-12)
    assert (wp >= 0).all()


def test_heavier_in_edges_lift_rank() -> None:
    """Steering more out-weight toward node 2 should raise its rank.

    Hub node 0 splits its out-mass between 1 and 2. With a 9:1 split toward 2,
    node 2 must outrank node 1.
    """
    a = np.array(
        [
            [0.0, 1.0, 9.0],  # node 0 sends 9x more weight to 2 than to 1
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    wp = weighted_pagerank(a, damping=0.85)
    assert wp[2] > wp[1]


# --- personalized PageRank -------------------------------------------------


def test_uniform_restart_equals_plain() -> None:
    """A uniform restart vector reproduces plain PageRank exactly."""
    restart = np.ones(3) / 3.0
    pp = personalized_pagerank(THREE_CYCLE, restart=restart, damping=0.85)
    plain = pagerank(THREE_CYCLE, damping=0.85)
    assert pp == pytest.approx(plain, abs=1e-9)


def test_restart_unnormalized_is_normalized() -> None:
    """Restart weights need not sum to 1; they are normalised internally."""
    a = _path(4)
    pp_norm = personalized_pagerank(a, restart=np.array([1.0, 0.0, 0.0, 0.0]))
    pp_raw = personalized_pagerank(a, restart=np.array([5.0, 0.0, 0.0, 0.0]))
    assert pp_norm == pytest.approx(pp_raw, abs=1e-12)


def test_one_hot_restart_concentrates_mass() -> None:
    """Restarting only at node 0 of a path biases mass toward that end.

    On the path 0--1--2--3 the restart sits on node 0. Among the nodes 0, 2, 3
    (all reached only through node 1) the personalized mass must decay strictly
    with distance from the restart node, and node 0 must dominate the far end 3.
    Node 1 can exceed node 0 because every path-walk funnels through it; that is
    a property of the chain, not a counter-example to the restart bias.
    """
    a = _path(4)
    pp = personalized_pagerank(a, restart=np.array([1.0, 0.0, 0.0, 0.0]))
    assert float(pp.sum()) == pytest.approx(1.0, abs=1e-12)
    assert pp[0] > pp[2] > pp[3]  # decays with distance from the restart node
    assert pp[0] > pp[3]


def test_personalized_invalid_inputs_raise() -> None:
    a = THREE_CYCLE
    with pytest.raises(ValueError):
        personalized_pagerank(a, restart=np.ones(2))  # wrong length
    with pytest.raises(ValueError):
        personalized_pagerank(a, restart=np.zeros(3))  # sums to zero
    with pytest.raises(ValueError):
        personalized_pagerank(a, restart=np.array([-1.0, 1.0, 1.0]))  # negative
    with pytest.raises(ValueError):
        personalized_pagerank(np.zeros((2, 3)), restart=np.ones(2))  # not square


# --- betweenness centrality ------------------------------------------------


def test_star_centre_betweenness_hand_value() -> None:
    """5-node star: hub betweenness = (n-1)(n-2)/2 = 6; leaves = 0."""
    n = 5
    bc = betweenness_centrality(_star(n))
    assert bc[0] == pytest.approx((n - 1) * (n - 2) / 2.0)
    assert bc[1:] == pytest.approx(np.zeros(n - 1))


def test_path_endpoints_zero_middle_positive() -> None:
    """3-path 0--1--2: endpoints 0, middle node lies between the ends (=1)."""
    bc = betweenness_centrality(_path(3))
    assert bc[0] == pytest.approx(0.0)
    assert bc[2] == pytest.approx(0.0)
    assert bc[1] == pytest.approx(1.0)


def test_betweenness_normalized_star() -> None:
    """The star hub lies on every leaf-pair shortest path => normalised 1.0."""
    bc = betweenness_centrality(_star(5), normalized=True)
    assert bc[0] == pytest.approx(1.0)
    assert bc[1:] == pytest.approx(np.zeros(4))


def test_betweenness_clique_all_zero() -> None:
    """In K4 every pair is adjacent, so no node sits between any other pair."""
    k4 = np.ones((4, 4)) - np.eye(4)
    bc = betweenness_centrality(k4)
    assert bc == pytest.approx(np.zeros(4))


def test_betweenness_single_node() -> None:
    assert betweenness_centrality(np.zeros((1, 1))) == pytest.approx([0.0])


def test_betweenness_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        betweenness_centrality(np.zeros((2, 3)))
