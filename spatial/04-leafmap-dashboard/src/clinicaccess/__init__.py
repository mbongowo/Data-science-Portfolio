"""clinicaccess: straight-line clinic-access screening for Cameroon.

A small, reproducible core that turns two point tables -- populated places
(lat/lon/population) and health facilities (lat/lon) -- into a quick picture of
who lives farthest from care. For every place it finds the great-circle
(haversine) distance to the nearest facility, summarises population coverage
within a set of distance thresholds, and ranks the most underserved places.

The distances here are straight-line, not road travel time. That makes this a
fast interactive screening tool, not a routing study; the sibling
``access-to-care`` project computes rigorous road-network travel time and the
two complement each other. See the README for the honest comparison.

The numeric core (:mod:`clinicaccess.distance`, :mod:`clinicaccess.access`) is
pure numpy/pandas so it can be unit-tested without any geospatial or web
dependency. The Streamlit/leafmap dashboard lives in ``app/`` and is never
imported by the tests.
"""

from __future__ import annotations

from clinicaccess.access import (
    coverage_stats,
    distance_bins,
    farthest_places,
    nearest_facility,
)
from clinicaccess.distance import haversine_km

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "coverage_stats",
    "distance_bins",
    "farthest_places",
    "haversine_km",
    "nearest_facility",
]
