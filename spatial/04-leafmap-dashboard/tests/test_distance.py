"""Known-answer tests for the haversine great-circle distance."""

from __future__ import annotations

import numpy as np

from clinicaccess.distance import EARTH_RADIUS_KM, haversine_km


def test_identical_points_zero():
    assert haversine_km(3.87, 11.52, 3.87, 11.52) == 0.0


def test_one_degree_latitude_at_equator():
    # One degree of latitude is ~111 km anywhere; check at the equator.
    d = float(haversine_km(0.0, 0.0, 1.0, 0.0))
    expected = EARTH_RADIUS_KM * np.radians(1.0)  # ~111.19 km
    assert abs(d - expected) < 1e-6
    assert abs(d - 111.19) < 0.1


def test_known_city_pair_yaounde_douala():
    # Yaounde (3.848, 11.502) to Douala (4.051, 9.768): ~195-200 km apart.
    d = float(haversine_km(3.848, 11.502, 4.051, 9.768))
    assert 190.0 < d < 205.0


def test_symmetry():
    a = float(haversine_km(3.848, 11.502, 4.051, 9.768))
    b = float(haversine_km(4.051, 9.768, 3.848, 11.502))
    assert abs(a - b) < 1e-9


def test_vectorised_broadcasting_shape():
    places_lat = np.array([[0.0], [1.0], [2.0]])  # (3, 1)
    places_lon = np.array([[0.0], [0.0], [0.0]])
    fac_lat = np.array([[0.0, 5.0]])  # (1, 2)
    fac_lon = np.array([[0.0, 0.0]])
    d = haversine_km(places_lat, places_lon, fac_lat, fac_lon)
    assert d.shape == (3, 2)
    assert d[0, 0] == 0.0  # first place sits on first facility
