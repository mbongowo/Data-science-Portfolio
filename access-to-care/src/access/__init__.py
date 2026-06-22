"""access-to-care: travel-time-to-services accessibility and equity analysis.

A small reproducible pipeline that turns a road network and health-facility
locations into population-weighted travel-time statistics by admin-2 unit.
Developed for a Cameroon study area; data sources are configurable.
"""

from __future__ import annotations

from access.access import (
    assign_nearest_source,
    graph_to_adjacency,
    nearest_facility_times,
    nearest_times_to_minutes,
    seconds_to_minutes,
)
from access.demo import run_demo
from access.equity import (
    aggregate_admins_to_national,
    coverage_bands,
    national_summary,
    population_within_thresholds,
    summarise_by_admin,
)
from access.metrics import (
    facility_load,
    gini_coefficient,
    two_step_floating_catchment,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "aggregate_admins_to_national",
    "assign_nearest_source",
    "coverage_bands",
    "facility_load",
    "gini_coefficient",
    "graph_to_adjacency",
    "national_summary",
    "nearest_facility_times",
    "nearest_times_to_minutes",
    "population_within_thresholds",
    "run_demo",
    "seconds_to_minutes",
    "summarise_by_admin",
    "two_step_floating_catchment",
]
