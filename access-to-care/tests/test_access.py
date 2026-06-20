"""Spot-check the routing and travel-time logic on tiny synthetic data.

These tests use only pure-Python helpers, so they run without any geospatial
dependency installed.
"""

from __future__ import annotations

import math

import pytest

from access.access import (
    graph_to_adjacency,
    nearest_facility_times,
    nearest_times_to_minutes,
    seconds_to_minutes,
)
from access.network import (
    assign_edge_speeds_and_times,
    speed_for_highway,
    total_network_time,
    travel_time_seconds,
)


def test_travel_time_seconds_basic() -> None:
    # 1000 m at 36 km/h (= 10 m/s) -> 100 s.
    assert travel_time_seconds(1000.0, 36.0) == pytest.approx(100.0)


def test_travel_time_rejects_nonpositive_speed() -> None:
    with pytest.raises(ValueError):
        travel_time_seconds(100.0, 0.0)


def test_speed_for_highway_lookup_and_default() -> None:
    speeds = {"primary": 70.0, "residential": 30.0}
    assert speed_for_highway("primary", speeds, 25.0) == 70.0
    assert speed_for_highway("unknown_tag", speeds, 25.0) == 25.0
    # A list of tags -> fastest matching speed.
    assert speed_for_highway(["residential", "primary"], speeds, 25.0) == 70.0


def test_assign_edge_speeds_and_times() -> None:
    speeds = {"primary": 36.0}  # 10 m/s
    edges = [{"length": 500.0, "highway": "primary"}]
    out = assign_edge_speeds_and_times(edges, speeds, default_speed_kph=18.0)
    assert out[0]["speed_kph"] == 36.0
    assert out[0]["travel_time"] == pytest.approx(50.0)
    # input not mutated
    assert "travel_time" not in edges[0]


def _diamond_adjacency() -> dict[str, list[tuple[str, float]]]:
    # A -> B (5), A -> C (2), C -> B (1), B -> D (1), C -> D (7)
    return {
        "A": [("B", 5.0), ("C", 2.0)],
        "B": [("D", 1.0)],
        "C": [("B", 1.0), ("D", 7.0)],
        "D": [],
    }


def test_single_source_dijkstra_shortest_paths() -> None:
    dist = nearest_facility_times(_diamond_adjacency(), ["A"])
    assert dist["A"] == 0.0
    assert dist["C"] == 2.0
    assert dist["B"] == 3.0  # A->C->B beats A->B
    assert dist["D"] == 4.0  # A->C->B->D


def test_multi_source_picks_nearest_facility() -> None:
    # Two facilities at A and D; every node's cost is to the closer one.
    dist = nearest_facility_times(_diamond_adjacency(), ["A", "D"])
    assert dist["A"] == 0.0
    assert dist["D"] == 0.0
    assert dist["C"] == 2.0  # closer to A
    # B is reachable from A (cost 3) but not from D (no outgoing path back); 3.
    assert dist["B"] == 3.0


def test_unreachable_nodes_omitted() -> None:
    adjacency = {"X": [], "Y": [("Z", 1.0)], "Z": []}
    dist = nearest_facility_times(adjacency, ["X"])
    assert dist == {"X": 0.0}
    assert "Y" not in dist


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError):
        nearest_facility_times({"A": [("B", -1.0)], "B": []}, ["A"])


def _two_facility_grid() -> dict[int, list[tuple[int, float]]]:
    # A 6-node line graph: 0-1-2-3-4-5, each hop 10 min, bidirectional.
    # Facilities sit at nodes 0 and 5, so every node routes to its nearer end.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    adj: dict[int, list[tuple[int, float]]] = {n: [] for n in range(6)}
    for u, v in edges:
        adj[u].append((v, 10.0))
        adj[v].append((u, 10.0))
    return adj


def test_multi_source_two_facilities_each_node_nearest() -> None:
    dist = nearest_facility_times(_two_facility_grid(), [0, 5])
    # Nodes split at the midpoint: 0,1,2 reach facility 0; 3,4,5 reach facility 5.
    assert dist[0] == 0.0
    assert dist[1] == 10.0
    assert dist[2] == 20.0
    assert dist[3] == 20.0  # nearer to facility at node 5
    assert dist[4] == 10.0
    assert dist[5] == 0.0


def test_seconds_to_minutes_and_unreachable() -> None:
    assert seconds_to_minutes(120.0) == pytest.approx(2.0)
    assert seconds_to_minutes(0.0) == pytest.approx(0.0)
    assert math.isnan(seconds_to_minutes(math.nan))
    assert math.isnan(seconds_to_minutes(math.inf))


def test_nearest_times_to_minutes_handles_missing_node() -> None:
    # node 2 is absent from the times dict -> unreachable -> NaN minutes.
    times = {10: 60.0, 11: 600.0}
    out = nearest_times_to_minutes([10, 11, 2], times)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(10.0)
    assert math.isnan(out[2])


def test_unreachable_demand_point_yields_nan() -> None:
    # A facility on one island; a demand node on a disconnected island.
    adjacency = {"f": [("a", 5.0)], "a": [("f", 5.0)], "island": []}
    times = nearest_facility_times(adjacency, ["f"])
    minutes = nearest_times_to_minutes(["a", "island"], times)
    assert minutes[0] == pytest.approx(5.0 / 60.0)
    assert math.isnan(minutes[1])  # island never reaches a facility


def test_total_network_time_sums_edges() -> None:
    edges = assign_edge_speeds_and_times(
        [{"length": 360.0, "highway": "primary"}, {"length": 360.0, "highway": "primary"}],
        {"primary": 36.0},  # 10 m/s -> 36 s per edge
        default_speed_kph=18.0,
    )
    assert total_network_time(edges) == pytest.approx(72.0)
    assert total_network_time([]) == 0.0


def test_graph_to_adjacency_matches_manual() -> None:
    class FakeGraph:
        """Minimal stand-in for a networkx graph (no networkx needed)."""

        def __init__(self) -> None:
            self.nodes = ["A", "B"]
            self._edges = [("A", "B", {"travel_time": 4.0})]

        def edges(self, data: bool = False):
            return self._edges if data else [(u, v) for u, v, _ in self._edges]

    adjacency = graph_to_adjacency(FakeGraph())
    assert adjacency == {"A": [("B", 4.0)], "B": []}
    dist = nearest_facility_times(adjacency, ["A"])
    assert dist["B"] == pytest.approx(4.0)
    assert math.isfinite(dist["B"])
