"""Inequality and spatial-accessibility metrics for travel-time-to-care.

Three reviewer-expected measures, each pure numpy/pandas so they are unit-tested
on tiny hand-checkable inputs without any geospatial dependency:

* :func:`gini_coefficient` -- population-weighted Gini of travel times, the
  standard 0 (perfect equality) to 1 (maximal inequality) inequality index.
* :func:`two_step_floating_catchment` -- the classic two-step floating catchment
  area (2SFCA) accessibility score per demand point.
* :func:`facility_load` -- the catchment demand (population) assigned to each
  facility, given a nearest-facility assignment.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


def gini_coefficient(
    values: Sequence[float] | np.ndarray | pd.Series,
    weights: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> float:
    """Population-weighted Gini coefficient of a set of values.

    The Gini coefficient measures inequality on a 0--1 scale: 0 means every unit
    has the same value (perfect equality), larger values mean more concentrated
    inequality. Applied to travel times, it summarises how unevenly access is
    distributed across the population.

    Formula (weighted, on values sorted ascending with weights ``w_i`` and values
    ``x_i``)::

        G = (sum_i sum_j w_i w_j |x_i - x_j|) / (2 * (sum_i w_i)^2 * mean_w)

    where ``mean_w = (sum_i w_i x_i) / (sum_i w_i)`` is the weighted mean. With
    equal weights this reduces to the standard unweighted Gini. Values must be
    non-negative for the index to lie in ``[0, 1]``.

    Parameters
    ----------
    values : array-like of float
        The quantity whose inequality is measured (e.g. travel time in minutes).
        Non-finite entries (NaN/inf) are dropped together with their weights.
    weights : array-like of float, optional
        Non-negative weights (e.g. population). Defaults to equal weights.

    Returns
    -------
    float
        The Gini coefficient in ``[0, 1]``. Returns 0.0 when there is no spread
        (all values equal, a single unit, or zero total weight).
    """
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    if x.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    finite = np.isfinite(x) & np.isfinite(w)
    x = x[finite]
    w = w[finite]
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non-negative for a Gini in [0, 1]")

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    total_w = w.sum()
    weighted_mean = float(np.sum(w * x)) / total_w if total_w > 0 else 0.0
    if total_w <= 0 or weighted_mean <= 0:
        return 0.0

    # Pairwise sum of w_i w_j |x_i - x_j| via the sorted closed form:
    # for sorted x, sum_{i<j} w_i w_j (x_j - x_i) is computed with prefix sums.
    cum_w = np.cumsum(w)
    cum_wx = np.cumsum(w * x)
    # For each j, contribution = w_j * (x_j * cum_w[j-1] - cum_wx[j-1]).
    prev_w = np.concatenate(([0.0], cum_w[:-1]))
    prev_wx = np.concatenate(([0.0], cum_wx[:-1]))
    pair_sum = float(np.sum(w * (x * prev_w - prev_wx)))

    # pair_sum = sum_{i<j} w_i w_j (x_j - x_i). The full weighted mean absolute
    # difference double-sum is 2*pair_sum, and G = MAD / (2 * mean), so the
    # factors of 2 cancel: G = pair_sum / (total_w^2 * mean).
    gini = pair_sum / (total_w**2 * weighted_mean)
    return float(gini)


def two_step_floating_catchment(
    demand_pop: Sequence[float] | np.ndarray,
    facility_capacity: Sequence[float] | np.ndarray,
    travel_times: np.ndarray,
    catchment_min: float,
) -> np.ndarray:
    """Two-step floating catchment area (2SFCA) accessibility score per demand point.

    The 2SFCA method (Luo & Wang, 2003) combines supply and demand within a
    travel-time catchment in two steps:

    **Step 1 -- supply-to-demand ratio at each facility.** For facility ``j`` with
    capacity ``S_j``, sum the population of every demand point ``k`` inside the
    catchment (``travel_time_kj <= catchment_min``) and form the ratio::

        R_j = S_j / sum_{k in catchment(j)} P_k

    **Step 2 -- accessibility at each demand point.** For demand point ``i``, sum
    the ratios of every facility ``j`` reachable within the catchment::

        A_i = sum_{j in catchment(i)} R_j

    A larger ``A_i`` means better access (more capacity per person reachable).
    Demand points with no facility in range score 0.

    Parameters
    ----------
    demand_pop : array-like of float, shape (n_demand,)
        Population at each demand point.
    facility_capacity : array-like of float, shape (n_facility,)
        Supply capacity at each facility (e.g. beds, staff, or 1.0 per facility).
    travel_times : numpy.ndarray, shape (n_demand, n_facility)
        Travel time from each demand point to each facility. Non-finite entries
        are treated as out of range.
    catchment_min : float
        Catchment threshold; a demand-facility pair is in range when its travel
        time is ``<= catchment_min``.

    Returns
    -------
    numpy.ndarray, shape (n_demand,)
        The 2SFCA accessibility score per demand point.
    """
    pop = np.asarray(demand_pop, dtype=float)
    cap = np.asarray(facility_capacity, dtype=float)
    tt = np.asarray(travel_times, dtype=float)
    if tt.shape != (pop.size, cap.size):
        raise ValueError(
            f"travel_times shape {tt.shape} must be (n_demand={pop.size}, n_facility={cap.size})"
        )

    in_range = np.isfinite(tt) & (tt <= catchment_min)

    # Step 1: supply-to-demand ratio R_j for each facility.
    pop_in_catchment = np.where(in_range, pop[:, None], 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(pop_in_catchment > 0, cap / pop_in_catchment, 0.0)

    # Step 2: sum reachable ratios at each demand point.
    access = np.where(in_range, ratios[None, :], 0.0).sum(axis=1)
    return access


def facility_load(
    node_ids: Sequence[Hashable],
    assigned_facility: Sequence[Hashable],
    population: Sequence[float] | np.ndarray | pd.Series,
) -> dict[Hashable, float]:
    """Catchment demand (total population) assigned to each facility.

    Given, per demand node, which facility it was assigned to (its nearest
    source) and its population, sum the population routed to each facility. This
    is the demand-side load each facility must serve.

    Demand nodes with no assigned facility (``assigned_facility`` is ``None`` or
    NaN, e.g. unreachable cells) are skipped and contribute to no facility.

    Parameters
    ----------
    node_ids : sequence of hashable
        Demand node ids (used only for length/alignment checks).
    assigned_facility : sequence of hashable
        Nearest facility id per demand node; ``None``/NaN marks unassigned.
    population : array-like of float
        Population per demand node, aligned with ``node_ids``.

    Returns
    -------
    dict of hashable to float
        ``facility id -> total assigned population``. Facilities that receive no
        demand do not appear.
    """
    node_ids = list(node_ids)
    assigned = list(assigned_facility)
    pop = np.asarray(population, dtype=float)
    if not (len(node_ids) == len(assigned) == pop.size):
        raise ValueError("node_ids, assigned_facility and population must align")

    load: dict[Hashable, float] = {}
    for fac, p in zip(assigned, pop, strict=True):
        if fac is None or (isinstance(fac, float) and np.isnan(fac)):
            continue
        if not np.isfinite(p):
            continue
        load[fac] = load.get(fac, 0.0) + float(p)
    return load
