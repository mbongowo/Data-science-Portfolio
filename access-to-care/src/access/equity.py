"""Population-weighted accessibility (equity) statistics.

Pipeline role: join nearest-facility travel times to WorldPop population and
summarise, per GADM admin-2 unit, how many people live within 30/60/120 minutes
of care and the share left beyond each threshold, then roll up to a national
figure.

The core arithmetic (:func:`population_within_thresholds`,
:func:`summarise_by_admin`, :func:`national_summary`) is implemented as pure
pandas/numpy functions so the shares can be verified (they sum to 1, thresholds
are correct) without any geospatial dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def population_within_thresholds(
    travel_time_min: Sequence[float] | np.ndarray | pd.Series,
    population: Sequence[float] | np.ndarray | pd.Series,
    thresholds_min: Sequence[float],
) -> dict[str, float]:
    """Population reachable within each threshold and the share beyond the largest.

    Returns a dict with, for each threshold ``t``:
      * ``pop_within_{t}min``  -- population with travel time <= t
      * ``share_within_{t}min`` -- that population / total population
      * ``share_beyond_{t}min`` -- 1 - share_within (NaN times count as beyond)

    Plus ``population_total``. Pure numpy; no geo dependency.
    """
    tt = np.asarray(travel_time_min, dtype=float)
    pop = np.asarray(population, dtype=float)
    if tt.shape != pop.shape:
        raise ValueError("travel_time_min and population must have the same shape")

    total = float(np.nansum(pop))
    result: dict[str, float] = {"population_total": total}

    for t in sorted(thresholds_min):
        within_mask = tt <= t  # NaN <= t is False, so unreachable counts as beyond
        pop_within = float(np.nansum(np.where(within_mask, pop, 0.0)))
        share_within = pop_within / total if total > 0 else float("nan")
        result[f"pop_within_{int(t)}min"] = pop_within
        result[f"share_within_{int(t)}min"] = share_within
        result[f"share_beyond_{int(t)}min"] = 1.0 - share_within if total > 0 else float("nan")
    return result


def summarise_by_admin(
    df: pd.DataFrame,
    thresholds_min: Sequence[float],
    admin_col: str = "admin2",
    time_col: str = "travel_time_min",
    pop_col: str = "population",
) -> pd.DataFrame:
    """Per-admin population-weighted accessibility table.

    ``df`` is a tidy frame of demand records: one row per demand cell with its
    admin unit, travel time (minutes), and population. Returns one row per admin
    unit with population totals, within/beyond shares per threshold. Pure pandas.
    """
    required = {admin_col, time_col, pop_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df is missing columns: {sorted(missing)}")

    rows: list[dict[str, float | str]] = []
    for admin_value, group in df.groupby(admin_col, sort=True):
        stats = population_within_thresholds(
            group[time_col].to_numpy(),
            group[pop_col].to_numpy(),
            thresholds_min,
        )
        row: dict[str, float | str] = {admin_col: admin_value}
        row.update(stats)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def national_summary(
    df: pd.DataFrame,
    thresholds_min: Sequence[float],
    time_col: str = "travel_time_min",
    pop_col: str = "population",
) -> dict[str, float]:
    """National (study-area-wide) population-weighted figure across all demand."""
    return population_within_thresholds(
        df[time_col].to_numpy(), df[pop_col].to_numpy(), thresholds_min
    )


def aggregate_admins_to_national(
    admin_summary: pd.DataFrame,
    thresholds_min: Sequence[float],
) -> dict[str, float]:
    """Roll up a per-admin summary table into one national figure.

    Takes the output of :func:`summarise_by_admin` and recombines it without
    touching the original demand cells. Population totals and within-threshold
    populations add across admin units; the national shares are recomputed from
    those sums. The result matches :func:`national_summary` computed directly on
    the demand frame, which is the property the equity numbers rely on.

    Parameters
    ----------
    admin_summary : pandas.DataFrame
        One row per admin unit, as returned by :func:`summarise_by_admin`. Must
        carry ``population_total`` and a ``pop_within_{t}min`` column per
        threshold.
    thresholds_min : sequence of float
        The same thresholds passed to :func:`summarise_by_admin`.

    Returns
    -------
    dict of str to float
        Same keys as :func:`population_within_thresholds`.
    """
    total = float(admin_summary["population_total"].sum())
    result: dict[str, float] = {"population_total": total}
    for t in sorted(thresholds_min):
        col = f"pop_within_{int(t)}min"
        if col not in admin_summary.columns:
            raise KeyError(f"admin_summary is missing column {col!r}")
        pop_within = float(admin_summary[col].sum())
        share_within = pop_within / total if total > 0 else float("nan")
        result[col] = pop_within
        result[f"share_within_{int(t)}min"] = share_within
        result[f"share_beyond_{int(t)}min"] = 1.0 - share_within if total > 0 else float("nan")
    return result


def coverage_bands(
    travel_time_min: Sequence[float] | np.ndarray | pd.Series,
    population: Sequence[float] | np.ndarray | pd.Series,
    thresholds_min: Sequence[float],
) -> dict[str, float]:
    """Population in each disjoint travel-time band, including an unreachable band.

    Where :func:`population_within_thresholds` reports cumulative coverage
    (population within each threshold), this splits the population into
    non-overlapping bands so they partition the total exactly. For thresholds
    ``[30, 60]`` the bands are ``0-30``, ``30-60``, ``60+`` (everyone past the
    last finite threshold) and ``unreachable`` (NaN/inf travel time).

    Parameters
    ----------
    travel_time_min : array-like of float
        Travel time per demand cell, in minutes. NaN or inf marks an
        unreachable cell.
    population : array-like of float
        Population per demand cell, aligned with ``travel_time_min``.
    thresholds_min : sequence of float
        Upper edges of the finite bands, in minutes.

    Returns
    -------
    dict of str to float
        ``pop_band_{lo}_{hi}min`` for each finite band, ``pop_band_{last}min_plus``
        for the open-ended band, ``pop_unreachable``, and ``population_total``.
        The band values sum to ``population_total``.
    """
    tt = np.asarray(travel_time_min, dtype=float)
    pop = np.asarray(population, dtype=float)
    if tt.shape != pop.shape:
        raise ValueError("travel_time_min and population must have the same shape")

    total = float(np.nansum(pop))
    result: dict[str, float] = {"population_total": total}

    reachable = np.isfinite(tt)
    unreachable_mask = ~reachable
    result["pop_unreachable"] = float(np.nansum(np.where(unreachable_mask, pop, 0.0)))

    edges = sorted(thresholds_min)
    lower = 0.0
    for hi in edges:
        # The first band includes its lower edge so a demand cell sitting exactly
        # on a facility (travel time 0) is counted; later bands are (lower, hi].
        above_lower = tt >= lower if lower == 0.0 else tt > lower
        band_mask = reachable & above_lower & (tt <= hi)
        result[f"pop_band_{int(lower)}_{int(hi)}min"] = float(
            np.nansum(np.where(band_mask, pop, 0.0))
        )
        lower = hi

    plus_mask = reachable & (tt > lower)
    result[f"pop_band_{int(lower)}min_plus"] = float(np.nansum(np.where(plus_mask, pop, 0.0)))
    return result


def attach_population(demand, population_raster: str, pop_col: str = "population"):
    """Sample the WorldPop raster onto demand points (zonal point sampling).

    Requires ``rasterstats``/``rasterio``. For point geometries this performs a
    point query; for the regular-grid demand surface each point represents one
    cell, so the sampled density is multiplied by the cell area handled upstream.
    """
    from rasterstats import point_query

    geo = demand.to_crs("EPSG:4326")
    values = point_query(geo.geometry, population_raster, interpolate="nearest")
    out = demand.copy()
    out[pop_col] = [float(v) if v is not None else 0.0 for v in values]
    return out


def join_admin(demand, admin):
    """Spatial-join demand points to admin-2 polygons; adds an ``admin2`` column."""
    import geopandas as gpd

    admin_g = admin.to_crs(demand.crs)
    name_col = next(
        (c for c in ("NAME_2", "name_2", "admin2", "ADM2_EN") if c in admin_g.columns),
        admin_g.columns[0],
    )
    joined = gpd.sjoin(demand, admin_g[[name_col, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={name_col: "admin2"}).drop(columns=["index_right"])
    return joined
