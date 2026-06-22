"""Great-circle distance, pure numpy.

The haversine formula gives the shortest distance over the Earth's surface
between two (lat, lon) points, treating the Earth as a sphere. For two points
with latitudes ``phi1, phi2`` and longitude difference ``dlambda``::

    a = sin^2(dphi / 2) + cos(phi1) * cos(phi2) * sin^2(dlambda / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c

where ``dphi = phi2 - phi1`` and ``R`` is the Earth's mean radius. We use
``R = 6371 km``. The function is vectorised and broadcasts, so it can take
scalars or arrays of any broadcast-compatible shapes -- which is what the
nearest-facility brute force relies on (places x facilities).
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two points (or arrays).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array-like
        Latitudes and longitudes in decimal degrees. Any broadcast-compatible
        shapes are allowed; the result takes the broadcast shape.

    Returns
    -------
    numpy.ndarray or float
        Distance(s) in kilometres, using Earth radius ``6371 km``. Identical
        points return ``0.0``.
    """
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # Clip guards against tiny floating-point excursions above 1.0 that would
    # make arcsin/sqrt return NaN for near-antipodal or identical points.
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c
