"""Verify the population-weighted equity arithmetic on synthetic data.

Pure pandas/numpy; no geospatial dependency required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from access.equity import (
    aggregate_admins_to_national,
    coverage_bands,
    national_summary,
    population_within_thresholds,
    summarise_by_admin,
)

THRESHOLDS = [30, 60, 120]


def _demand_frame() -> pd.DataFrame:
    # Two admin units, four demand cells each, known populations and times.
    return pd.DataFrame(
        {
            "admin2": ["North", "North", "North", "North", "South", "South", "South", "South"],
            "travel_time_min": [10.0, 45.0, 90.0, 200.0, 5.0, 25.0, 70.0, np.nan],
            "population": [100.0, 200.0, 300.0, 400.0, 50.0, 50.0, 100.0, 100.0],
        }
    )


def test_within_thresholds_basic_counts() -> None:
    tt = [10.0, 45.0, 90.0, 200.0]
    pop = [100.0, 200.0, 300.0, 400.0]
    stats = population_within_thresholds(tt, pop, THRESHOLDS)
    assert stats["population_total"] == pytest.approx(1000.0)
    assert stats["pop_within_30min"] == pytest.approx(100.0)  # only the 10-min cell
    assert stats["pop_within_60min"] == pytest.approx(300.0)  # 10 + 45 min cells
    assert stats["pop_within_120min"] == pytest.approx(600.0)  # +90 min cell


def test_within_and_beyond_shares_sum_to_one() -> None:
    stats = population_within_thresholds(
        [10.0, 45.0, 90.0, 200.0], [100.0, 200.0, 300.0, 400.0], THRESHOLDS
    )
    for t in THRESHOLDS:
        within = stats[f"share_within_{t}min"]
        beyond = stats[f"share_beyond_{t}min"]
        assert within + beyond == pytest.approx(1.0)
    # Shares are monotone non-decreasing as the threshold grows.
    assert (
        stats["share_within_30min"] <= stats["share_within_60min"] <= stats["share_within_120min"]
    )


def test_nan_times_count_as_beyond() -> None:
    # An unreachable (NaN) cell must never be counted as "within".
    stats = population_within_thresholds([np.nan, 10.0], [500.0, 500.0], [60])
    assert stats["population_total"] == pytest.approx(1000.0)
    assert stats["pop_within_60min"] == pytest.approx(500.0)
    assert stats["share_within_60min"] == pytest.approx(0.5)
    assert stats["share_beyond_60min"] == pytest.approx(0.5)


def test_summarise_by_admin_rows_and_population() -> None:
    out = summarise_by_admin(_demand_frame(), THRESHOLDS)
    assert set(out["admin2"]) == {"North", "South"}
    north = out.set_index("admin2").loc["North"]
    south = out.set_index("admin2").loc["South"]
    assert north["population_total"] == pytest.approx(1000.0)
    assert south["population_total"] == pytest.approx(300.0)
    # Every admin's within/beyond shares sum to 1 at each threshold.
    for _, row in out.iterrows():
        for t in THRESHOLDS:
            assert row[f"share_within_{t}min"] + row[f"share_beyond_{t}min"] == pytest.approx(1.0)


def test_national_total_equals_sum_of_admins() -> None:
    df = _demand_frame()
    per_admin = summarise_by_admin(df, THRESHOLDS)
    nat = national_summary(df, THRESHOLDS)
    assert nat["population_total"] == pytest.approx(per_admin["population_total"].sum())
    # National within-pop equals sum of per-admin within-pop at each threshold.
    for t in THRESHOLDS:
        col = f"pop_within_{t}min"
        assert nat[col] == pytest.approx(per_admin[col].sum())


def test_aggregate_admins_equals_direct_national() -> None:
    # Rolling up the per-admin table must reproduce the national figure computed
    # straight from the demand cells (population-weighted aggregation property).
    df = _demand_frame()
    per_admin = summarise_by_admin(df, THRESHOLDS)
    rolled = aggregate_admins_to_national(per_admin, THRESHOLDS)
    direct = national_summary(df, THRESHOLDS)
    assert rolled["population_total"] == pytest.approx(direct["population_total"])
    for t in THRESHOLDS:
        assert rolled[f"pop_within_{t}min"] == pytest.approx(direct[f"pop_within_{t}min"])
        assert rolled[f"share_within_{t}min"] == pytest.approx(direct[f"share_within_{t}min"])
        assert rolled[f"share_beyond_{t}min"] == pytest.approx(direct[f"share_beyond_{t}min"])


def test_aggregate_admins_missing_column_raises() -> None:
    bad = pd.DataFrame({"admin2": ["A"], "population_total": [10.0]})
    with pytest.raises(KeyError):
        aggregate_admins_to_national(bad, THRESHOLDS)


def test_coverage_bands_partition_total() -> None:
    # Bands plus the unreachable bucket must sum back to the population total.
    tt = [10.0, 45.0, 90.0, 200.0, np.nan]
    pop = [100.0, 200.0, 300.0, 400.0, 500.0]
    bands = coverage_bands(tt, pop, THRESHOLDS)
    assert bands["population_total"] == pytest.approx(1500.0)
    assert bands["pop_band_0_30min"] == pytest.approx(100.0)
    assert bands["pop_band_30_60min"] == pytest.approx(200.0)
    assert bands["pop_band_60_120min"] == pytest.approx(300.0)
    assert bands["pop_band_120min_plus"] == pytest.approx(400.0)
    assert bands["pop_unreachable"] == pytest.approx(500.0)
    band_sum = (
        bands["pop_band_0_30min"]
        + bands["pop_band_30_60min"]
        + bands["pop_band_60_120min"]
        + bands["pop_band_120min_plus"]
        + bands["pop_unreachable"]
    )
    assert band_sum == pytest.approx(bands["population_total"])


def test_coverage_bands_first_band_matches_cumulative() -> None:
    # The first band (0..t) equals cumulative coverage within the first threshold.
    tt = [10.0, 45.0, 90.0]
    pop = [100.0, 200.0, 300.0]
    bands = coverage_bands(tt, pop, THRESHOLDS)
    within = population_within_thresholds(tt, pop, THRESHOLDS)
    assert bands["pop_band_0_30min"] == pytest.approx(within["pop_within_30min"])


def test_coverage_bands_boundary_inclusive_upper() -> None:
    # A cell exactly on a threshold edge falls in the lower band (<= upper).
    bands = coverage_bands([30.0, 60.0], [10.0, 20.0], [30, 60])
    assert bands["pop_band_0_30min"] == pytest.approx(10.0)
    assert bands["pop_band_30_60min"] == pytest.approx(20.0)
    assert bands["pop_band_60min_plus"] == pytest.approx(0.0)


def test_coverage_bands_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        coverage_bands([1.0], [1.0, 2.0], THRESHOLDS)


def test_summarise_missing_column_raises() -> None:
    df = pd.DataFrame({"admin2": ["A"], "travel_time_min": [1.0]})
    with pytest.raises(KeyError):
        summarise_by_admin(df, THRESHOLDS)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        population_within_thresholds([1.0, 2.0], [1.0], THRESHOLDS)
