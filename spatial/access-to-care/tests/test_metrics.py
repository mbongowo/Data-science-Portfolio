"""Known-answer tests for inequality and accessibility metrics.

Every expected value here is derived by hand from the formula, so the tests pin
the maths rather than the implementation. Pure numpy/pandas; no geo stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from access.access import assign_nearest_source
from access.metrics import (
    facility_load,
    gini_coefficient,
    two_step_floating_catchment,
)

# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------


def test_gini_of_equal_values_is_zero() -> None:
    assert gini_coefficient([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_gini_single_value_is_zero() -> None:
    assert gini_coefficient([42.0]) == pytest.approx(0.0)


def test_gini_known_small_set() -> None:
    # Unweighted Gini of [1, 2, 3, 4]. Mean = 2.5.
    # Sum of |x_i - x_j| over all ordered pairs = 2 * sum_{i<j}(x_j - x_i)
    #   pairs (1,2)(1,3)(1,4)(2,3)(2,4)(3,4) -> diffs 1+2+3+1+2+1 = 10, doubled = 20.
    # G = 20 / (2 * n^2 * mean) = 20 / (2 * 16 * 2.5) = 20 / 80 = 0.25.
    assert gini_coefficient([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)


def test_gini_maximal_concentration() -> None:
    # All value held by one unit out of n: Gini -> 1 - 1/n. For n=4 -> 0.75.
    assert gini_coefficient([0.0, 0.0, 0.0, 12.0]) == pytest.approx(0.75)


def test_gini_weighted_equals_expanded_unweighted() -> None:
    # Weighting [1,2] with weights [2,1] equals the unweighted set [1,1,2].
    weighted = gini_coefficient([1.0, 2.0], weights=[2.0, 1.0])
    expanded = gini_coefficient([1.0, 1.0, 2.0])
    assert weighted == pytest.approx(expanded)


def test_gini_drops_nonfinite() -> None:
    # A NaN value (unreachable cell) is dropped with its weight, not counted.
    g = gini_coefficient([1.0, 2.0, 3.0, 4.0, np.nan], weights=[1, 1, 1, 1, 9])
    assert g == pytest.approx(0.25)


def test_gini_zero_weight_total_is_zero() -> None:
    assert gini_coefficient([1.0, 2.0, 3.0], weights=[0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_gini_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        gini_coefficient([-1.0, 2.0])


# ---------------------------------------------------------------------------
# Two-step floating catchment area (2SFCA)
# ---------------------------------------------------------------------------


def test_2sfca_tiny_toy() -> None:
    # 2 demand points, 2 facilities. catchment = 30 min.
    # travel times (min):
    #   d0: f0=10 (in), f1=40 (out)
    #   d1: f0=20 (in), f1=15 (in)
    # populations: d0=100, d1=300 ; capacities: f0=10, f1=20.
    # Step 1 ratios:
    #   f0 catchment = {d0, d1} pop=400 -> R0 = 10/400 = 0.025
    #   f1 catchment = {d1}     pop=300 -> R1 = 20/300 = 0.0666667
    # Step 2:
    #   A_d0 = R0          = 0.025
    #   A_d1 = R0 + R1     = 0.025 + 0.0666667 = 0.0916667
    tt = np.array([[10.0, 40.0], [20.0, 15.0]])
    access = two_step_floating_catchment(
        demand_pop=[100.0, 300.0],
        facility_capacity=[10.0, 20.0],
        travel_times=tt,
        catchment_min=30.0,
    )
    assert access[0] == pytest.approx(0.025)
    assert access[1] == pytest.approx(0.025 + 20.0 / 300.0)


def test_2sfca_out_of_range_scores_zero() -> None:
    # Single facility out of range for the only demand point -> access 0.
    tt = np.array([[45.0]])
    access = two_step_floating_catchment([100.0], [10.0], tt, catchment_min=30.0)
    assert access[0] == pytest.approx(0.0)


def test_2sfca_single_facility_in_range() -> None:
    # One facility serving both demand points within range.
    # pop in catchment = 100 + 100 = 200, cap = 50 -> R = 0.25; each A = 0.25.
    tt = np.array([[10.0], [20.0]])
    access = two_step_floating_catchment([100.0, 100.0], [50.0], tt, catchment_min=30.0)
    assert access[0] == pytest.approx(0.25)
    assert access[1] == pytest.approx(0.25)


def test_2sfca_nonfinite_time_is_out_of_range() -> None:
    tt = np.array([[np.inf, 10.0]])
    access = two_step_floating_catchment([100.0], [10.0, 10.0], tt, catchment_min=30.0)
    # only f1 in range, its catchment pop = 100 -> R1 = 0.1.
    assert access[0] == pytest.approx(0.1)


def test_2sfca_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        two_step_floating_catchment([1.0, 2.0], [1.0], np.zeros((1, 1)), 30.0)


# ---------------------------------------------------------------------------
# facility_load
# ---------------------------------------------------------------------------


def test_facility_load_sums_population_per_facility() -> None:
    nodes = [0, 1, 2, 3]
    assigned = ["A", "A", "B", "B"]
    pop = [100.0, 50.0, 200.0, 25.0]
    load = facility_load(nodes, assigned, pop)
    assert load == {"A": pytest.approx(150.0), "B": pytest.approx(225.0)}


def test_facility_load_skips_unassigned() -> None:
    load = facility_load([0, 1, 2], ["A", None, "A"], [10.0, 999.0, 5.0])
    assert load == {"A": pytest.approx(15.0)}
    assert 999.0 not in load.values()


def test_facility_load_single_facility() -> None:
    load = facility_load([0, 1], ["only", "only"], [3.0, 7.0])
    assert load == {"only": pytest.approx(10.0)}


def test_facility_load_zero_population() -> None:
    load = facility_load([0, 1], ["A", "B"], [0.0, 0.0])
    assert load == {"A": pytest.approx(0.0), "B": pytest.approx(0.0)}


def test_facility_load_misaligned_raises() -> None:
    with pytest.raises(ValueError):
        facility_load([0, 1], ["A"], [1.0, 2.0])


# ---------------------------------------------------------------------------
# assign_nearest_source (cost + which source), feeding facility_load
# ---------------------------------------------------------------------------


def _line_adjacency() -> dict[int, list[tuple[int, float]]]:
    # 0-1-2-3-4-5, each hop 10, bidirectional. Facilities at 0 and 5.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    adj: dict[int, list[tuple[int, float]]] = {n: [] for n in range(6)}
    for u, v in edges:
        adj[u].append((v, 10.0))
        adj[v].append((u, 10.0))
    return adj


def test_assign_nearest_source_cost_and_source() -> None:
    dist, nearest = assign_nearest_source(_line_adjacency(), [0, 5])
    # Costs match the plain Dijkstra; sources split at the midpoint.
    assert dist[2] == pytest.approx(20.0)
    assert dist[3] == pytest.approx(20.0)
    assert nearest[0] == 0
    assert nearest[1] == 0
    assert nearest[2] == 0  # nearer to facility 0
    assert nearest[3] == 5  # nearer to facility 5
    assert nearest[5] == 5


def test_assign_nearest_source_feeds_facility_load() -> None:
    _, nearest = assign_nearest_source(_line_adjacency(), [0, 5])
    nodes = list(range(6))
    population = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    assigned = [nearest[n] for n in nodes]
    load = facility_load(nodes, assigned, population)
    # Nodes 0,1,2 -> facility 0 (30); nodes 3,4,5 -> facility 5 (30).
    assert load[0] == pytest.approx(30.0)
    assert load[5] == pytest.approx(30.0)


def test_assign_nearest_source_all_unreachable_island() -> None:
    adjacency = {"f": [("a", 5.0)], "a": [("f", 5.0)], "island": []}
    dist, nearest = assign_nearest_source(adjacency, ["f"])
    assert "island" not in dist
    assert "island" not in nearest
    assert nearest["a"] == "f"
