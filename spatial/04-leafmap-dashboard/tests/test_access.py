"""Known-answer tests for nearest-facility, coverage, ranking and bins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinicaccess.access import (
    coverage_stats,
    distance_bins,
    farthest_places,
    nearest_facility,
)
from clinicaccess.distance import haversine_km


def test_nearest_facility_picks_obvious_closest():
    places = pd.DataFrame({"lat": [0.0, 10.0], "lon": [0.0, 0.0], "population": [100.0, 50.0]})
    facilities = pd.DataFrame({"facility_id": ["A", "B"], "lat": [0.0, 10.0], "lon": [0.0, 0.0]})
    out = nearest_facility(places, facilities)
    assert list(out["nearest_facility_id"]) == ["A", "B"]
    # Each place sits exactly on its facility -> zero distance.
    assert out["nearest_km"].tolist() == [0.0, 0.0]


def test_nearest_facility_exact_km():
    # Place one degree of latitude north of a single facility at the equator.
    places = pd.DataFrame({"lat": [1.0], "lon": [0.0]})
    facilities = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    out = nearest_facility(places, facilities)
    expected = float(haversine_km(1.0, 0.0, 0.0, 0.0))
    assert abs(float(out["nearest_km"].iloc[0]) - expected) < 1e-9
    # Falls back to positional index when no facility_id column.
    assert out["nearest_facility_id"].iloc[0] == 0


def test_nearest_facility_guards():
    good = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    with pytest.raises(KeyError):
        nearest_facility(pd.DataFrame({"x": [1]}), good)
    with pytest.raises(ValueError):
        nearest_facility(good.iloc[:0], good)
    with pytest.raises(ValueError):
        nearest_facility(good, good.iloc[:0])


def test_coverage_stats_known_shares():
    distances = np.array([2.0, 7.0, 20.0, 40.0])
    population = np.array([10.0, 10.0, 10.0, 10.0])  # total 40
    stats = coverage_stats(distances, population, [5, 10, 25])
    assert stats["population_total"] == 40.0
    assert stats["pop_within_5km"] == 10.0
    assert stats["share_within_5km"] == pytest.approx(0.25)
    assert stats["share_within_10km"] == pytest.approx(0.5)  # 2 and 7 km
    assert stats["share_within_25km"] == pytest.approx(0.75)  # +20 km
    assert stats["share_beyond_25km"] == pytest.approx(0.25)  # the 40 km one
    # Cumulative shares are non-decreasing with the threshold.
    assert stats["share_within_5km"] <= stats["share_within_10km"] <= stats["share_within_25km"]


def test_coverage_stats_nan_counts_as_beyond():
    distances = np.array([2.0, np.nan])
    population = np.array([10.0, 30.0])
    stats = coverage_stats(distances, population, [5])
    assert stats["population_total"] == 40.0
    assert stats["pop_within_5km"] == 10.0
    assert stats["share_beyond_5km"] == pytest.approx(0.75)


def test_coverage_stats_shape_guard():
    with pytest.raises(ValueError):
        coverage_stats([1.0, 2.0], [1.0], [5])


def test_farthest_places_returns_right_n_in_order():
    df = pd.DataFrame({"name": ["a", "b", "c", "d"], "nearest_km": [3.0, 30.0, 12.0, 1.0]})
    top = farthest_places(df, n=2)
    assert list(top["name"]) == ["b", "c"]
    assert list(top["nearest_km"]) == [30.0, 12.0]


def test_farthest_places_nan_sorts_last():
    df = pd.DataFrame({"name": ["a", "b"], "nearest_km": [np.nan, 5.0]})
    top = farthest_places(df, n=1)
    assert top["name"].iloc[0] == "b"


def test_distance_bins_labels():
    bins = distance_bins([0.0, 3.0, 7.0, 20.0, 40.0], [5, 10, 25])
    assert list(bins) == ["0-5 km", "0-5 km", "5-10 km", "10-25 km", "25+ km"]


def test_distance_bins_guards():
    with pytest.raises(ValueError):
        distance_bins([1.0], [])
    with pytest.raises(ValueError):
        distance_bins([1.0], [10, 5])
