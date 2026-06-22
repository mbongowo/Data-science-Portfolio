"""Known-answer tests for the dimensional-modelling helpers.

These check the two warehouse building blocks the dbt marts mirror in SQL:

* surrogate_key is deterministic (stable across calls) and distinct for
  distinct rows.
* build_date_dim has exactly one row per inclusive calendar day, with the
  expected fields on a hand-checked row.

No dbt / duckdb dependency, so they always execute.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dwh.dimensional import build_date_dim, surrogate_key


def _titles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002", "tt0000003"],
            "titleType": ["movie", "short", "short"],
        }
    )


def test_surrogate_key_is_deterministic() -> None:
    """Same inputs hash to the same keys on a second call."""
    df = _titles()
    k1 = surrogate_key(df, ["tconst", "titleType"])
    k2 = surrogate_key(df, ["tconst", "titleType"])
    assert list(k1) == list(k2)


def test_surrogate_key_distinct_for_distinct_rows() -> None:
    """Distinct rows get distinct keys."""
    df = _titles()
    keys = surrogate_key(df, ["tconst", "titleType"])
    assert keys.nunique() == len(df)


def test_surrogate_key_same_for_identical_rows() -> None:
    """Two rows with identical key columns collide on purpose (same entity)."""
    df = pd.DataFrame({"tconst": ["tt1", "tt1"], "titleType": ["movie", "movie"]})
    keys = surrogate_key(df, ["tconst", "titleType"])
    assert keys.iloc[0] == keys.iloc[1]


def test_surrogate_key_order_sensitive() -> None:
    """Column order is part of the identity: swapping columns changes the key."""
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    k_ab = surrogate_key(df, ["a", "b"]).iloc[0]
    k_ba = surrogate_key(df, ["b", "a"]).iloc[0]
    assert k_ab != k_ba


def test_surrogate_key_is_hex_sha256() -> None:
    """Keys are 64-char hex SHA-256 digests."""
    keys = surrogate_key(_titles(), ["tconst"])
    assert all(len(k) == 64 for k in keys)
    assert all(all(c in "0123456789abcdef" for c in k) for k in keys)


def test_surrogate_key_empty_cols_raises() -> None:
    with pytest.raises(ValueError):
        surrogate_key(_titles(), [])


def test_surrogate_key_missing_col_raises() -> None:
    with pytest.raises(ValueError):
        surrogate_key(_titles(), ["nope"])


def test_build_date_dim_row_count_is_inclusive() -> None:
    """January 2024 has 31 days; the inclusive range yields 31 rows."""
    dim = build_date_dim("2024-01-01", "2024-01-31")
    assert len(dim) == 31


def test_build_date_dim_single_day() -> None:
    """A start == end range is one row (inclusive of both endpoints)."""
    dim = build_date_dim("2024-02-29", "2024-02-29")  # leap day
    assert len(dim) == 1
    assert dim.loc[0, "year"] == 2024
    assert dim.loc[0, "month"] == 2
    assert dim.loc[0, "day"] == 29


def test_build_date_dim_known_row_fields() -> None:
    """2024-01-06 is a Saturday (dow=5, is_weekend=True)."""
    dim = build_date_dim("2024-01-01", "2024-01-31")
    row = dim[dim["date"] == pd.Timestamp("2024-01-06")].iloc[0]
    assert row["year"] == 2024
    assert row["month"] == 1
    assert row["day"] == 6
    assert row["dow"] == 5  # Saturday (Monday=0)
    assert bool(row["is_weekend"]) is True


def test_build_date_dim_weekday_is_not_weekend() -> None:
    """2024-01-03 is a Wednesday (dow=2, is_weekend=False)."""
    dim = build_date_dim("2024-01-01", "2024-01-31")
    row = dim[dim["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert row["dow"] == 2
    assert bool(row["is_weekend"]) is False


def test_build_date_dim_columns() -> None:
    dim = build_date_dim("2024-01-01", "2024-01-02")
    assert list(dim.columns) == [
        "date",
        "year",
        "month",
        "day",
        "dow",
        "is_weekend",
    ]


def test_build_date_dim_accepts_date_objects() -> None:
    dim = build_date_dim(date(2024, 1, 1), date(2024, 1, 10))
    assert len(dim) == 10


def test_build_date_dim_end_before_start_raises() -> None:
    with pytest.raises(ValueError):
        build_date_dim("2024-01-31", "2024-01-01")
