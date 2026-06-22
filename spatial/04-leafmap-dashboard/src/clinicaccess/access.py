"""Nearest-facility distance, coverage and ranking -- pure numpy/pandas.

Pipeline role: given a table of populated places (lat/lon/population) and a
table of health facilities (lat/lon), compute for every place the straight-line
distance to its nearest facility, then summarise population coverage within a
set of distance thresholds and rank the most underserved places. No geospatial
or web dependency, so the arithmetic is unit-testable.

The coverage semantics mirror the sibling ``access-to-care`` equity functions:
cumulative population within each threshold plus the share left beyond the
largest one, NaN-safe.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

LAT_LON = ("lat", "lon")


def _require_columns(df: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def nearest_facility(places: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """Distance to the nearest facility for every place (vectorised brute force).

    Parameters
    ----------
    places : pandas.DataFrame
        One row per populated place, with ``lat`` and ``lon`` columns.
    facilities : pandas.DataFrame
        One row per facility, with ``lat`` and ``lon`` columns. If a
        ``facility_id`` column is present it is used as the reported id;
        otherwise the facility's positional index is used.

    Returns
    -------
    pandas.DataFrame
        A copy of ``places`` with two added columns: ``nearest_km`` (great-circle
        distance to the closest facility) and ``nearest_facility_id`` (the id or
        index of that facility). The index is preserved.

    Raises
    ------
    KeyError
        If either frame is missing ``lat``/``lon``.
    ValueError
        If ``places`` or ``facilities`` is empty.
    """
    _require_columns(places, "places", LAT_LON)
    _require_columns(facilities, "facilities", LAT_LON)
    if len(places) == 0:
        raise ValueError("places is empty")
    if len(facilities) == 0:
        raise ValueError("facilities is empty")

    from clinicaccess.distance import haversine_km

    place_lat = places["lat"].to_numpy(dtype=float)[:, np.newaxis]
    place_lon = places["lon"].to_numpy(dtype=float)[:, np.newaxis]
    fac_lat = facilities["lat"].to_numpy(dtype=float)[np.newaxis, :]
    fac_lon = facilities["lon"].to_numpy(dtype=float)[np.newaxis, :]

    # (n_places, n_facilities) distance matrix, then reduce along facilities.
    dist = haversine_km(place_lat, place_lon, fac_lat, fac_lon)
    nearest_idx = np.argmin(dist, axis=1)
    nearest_km = dist[np.arange(dist.shape[0]), nearest_idx]

    if "facility_id" in facilities.columns:
        ids = facilities["facility_id"].to_numpy()[nearest_idx]
    else:
        ids = facilities.index.to_numpy()[nearest_idx]

    out = places.copy()
    out["nearest_km"] = nearest_km
    out["nearest_facility_id"] = ids
    return out


def coverage_stats(
    distances_km: Sequence[float] | np.ndarray | pd.Series,
    population: Sequence[float] | np.ndarray | pd.Series,
    thresholds_km: Sequence[float],
) -> dict[str, float]:
    """Population reachable within each distance threshold, plus the share beyond.

    Mirrors ``access-to-care``'s equity semantics, but on straight-line distance
    rather than travel time. For each threshold ``t`` (km):

      * ``pop_within_{t}km``   -- population whose nearest facility is <= t km
      * ``share_within_{t}km`` -- that population / total population
      * ``share_beyond_{t}km`` -- ``1 - share_within`` (NaN distances count as
        beyond)

    Plus ``population_total``. NaN-safe; a place with NaN distance is treated as
    beyond every threshold. Pure numpy.

    Raises
    ------
    ValueError
        If ``distances_km`` and ``population`` differ in shape.
    """
    dist = np.asarray(distances_km, dtype=float)
    pop = np.asarray(population, dtype=float)
    if dist.shape != pop.shape:
        raise ValueError("distances_km and population must have the same shape")

    total = float(np.nansum(pop))
    result: dict[str, float] = {"population_total": total}

    for t in sorted(thresholds_km):
        within_mask = dist <= t  # NaN <= t is False, so unreachable counts as beyond
        pop_within = float(np.nansum(np.where(within_mask, pop, 0.0)))
        share_within = pop_within / total if total > 0 else float("nan")
        key = _km_key(t)
        result[f"pop_within_{key}km"] = pop_within
        result[f"share_within_{key}km"] = share_within
        result[f"share_beyond_{key}km"] = 1.0 - share_within if total > 0 else float("nan")
    return result


def _km_key(t: float) -> str:
    """Format a threshold for a dict key: integers stay clean (5, 10, 25)."""
    return str(int(t)) if float(t).is_integer() else str(t)


def farthest_places(places_with_dist: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` places with the largest ``nearest_km`` -- the underserved ones.

    Parameters
    ----------
    places_with_dist : pandas.DataFrame
        Output of :func:`nearest_facility` (must carry ``nearest_km``).
    n : int
        How many of the farthest places to return.

    Returns
    -------
    pandas.DataFrame
        Up to ``n`` rows sorted by ``nearest_km`` descending. NaN distances sort
        last. The original columns and index are preserved.
    """
    _require_columns(places_with_dist, "places_with_dist", ("nearest_km",))
    if n < 0:
        raise ValueError("n must be non-negative")
    ordered = places_with_dist.sort_values("nearest_km", ascending=False, na_position="last")
    return ordered.head(n)


def distance_bins(
    distances_km: Sequence[float] | np.ndarray | pd.Series,
    edges: Sequence[float],
) -> pd.Categorical:
    """Categorical distance-band label per place (for graduated map colours).

    Bins are left-open, right-closed on the interior edges, with a leading
    ``0-{e0} km`` band that includes 0 and a trailing ``{last}+ km`` open band.
    For ``edges = [5, 10, 25]`` the labels are
    ``0-5 km``, ``5-10 km``, ``10-25 km``, ``25+ km``. NaN distances map to a
    missing category (``NaN``).

    Parameters
    ----------
    distances_km : array-like of float
        Nearest-facility distance per place.
    edges : sequence of float
        Increasing interior bin edges in km.

    Returns
    -------
    pandas.Categorical
        One ordered label per input distance.

    Raises
    ------
    ValueError
        If ``edges`` is empty or not strictly increasing.
    """
    dist = np.asarray(distances_km, dtype=float)
    edges = list(edges)
    if not edges:
        raise ValueError("edges must contain at least one edge")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError("edges must be strictly increasing")

    bin_edges = [-np.inf, *edges, np.inf]
    labels: list[str] = []
    labels.append(f"0-{_km_key(edges[0])} km")
    for lo, hi in zip(edges, edges[1:], strict=False):
        labels.append(f"{_km_key(lo)}-{_km_key(hi)} km")
    labels.append(f"{_km_key(edges[-1])}+ km")

    return pd.cut(
        dist,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
