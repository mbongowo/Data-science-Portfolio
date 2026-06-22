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

from dwh.dimensional import (
    SCD2_FAR_FUTURE,
    build_date_dim,
    scd2_snapshot,
    surrogate_key,
)


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


# --------------------------------------------------------------------------- #
# scd2_snapshot
# --------------------------------------------------------------------------- #
#
# Worked SCD2 example, tracking ``title_type`` per ``title_id``.
#
# Run 1 (valid_from = 2024-01-01), two titles:
#     tt1  movie
#     tt2  short
#   => two current rows, each [2024-01-01 .. 9999-12-31], is_current=True.
#
# Run 2 (valid_from = 2024-06-01):
#     tt1  tvSeries   (CHANGED from movie)
#     tt2  short      (unchanged)
#     tt3  documentary (NEW)
#   Expected snapshot, by hand (4 rows):
#     tt1 movie       2024-01-01 .. 2024-06-01   is_current=False
#     tt1 tvSeries    2024-06-01 .. 9999-12-31   is_current=True
#     tt2 short       2024-01-01 .. 9999-12-31   is_current=True   (carried forward)
#     tt3 documentary 2024-06-01 .. 9999-12-31   is_current=True


def _run1() -> pd.DataFrame:
    current = pd.DataFrame(
        {"title_id": ["tt1", "tt2"], "title_type": ["movie", "short"]}
    )
    return scd2_snapshot(
        current, None, key="title_id", cols=["title_type"], valid_from="2024-01-01"
    )


def test_scd2_first_run_all_current() -> None:
    snap = _run1()
    assert len(snap) == 2
    assert snap["is_current"].all()
    assert (snap["valid_from"] == pd.Timestamp("2024-01-01")).all()
    assert (snap["valid_to"] == SCD2_FAR_FUTURE).all()
    assert list(snap.columns) == [
        "title_id",
        "title_type",
        "valid_from",
        "valid_to",
        "is_current",
    ]


def test_scd2_change_expires_and_appends() -> None:
    prev = _run1()
    current = pd.DataFrame(
        {
            "title_id": ["tt1", "tt2", "tt3"],
            "title_type": ["tvSeries", "short", "documentary"],
        }
    )
    snap = scd2_snapshot(
        current, prev, key="title_id", cols=["title_type"], valid_from="2024-06-01"
    )

    # Exactly the four hand-derived rows.
    assert len(snap) == 4

    # tt1: one closed movie version + one current tvSeries version.
    tt1 = snap[snap["title_id"] == "tt1"].sort_values("valid_from")
    assert list(tt1["title_type"]) == ["movie", "tvSeries"]
    assert list(tt1["is_current"]) == [False, True]
    old, new = tt1.iloc[0], tt1.iloc[1]
    assert old["valid_from"] == pd.Timestamp("2024-01-01")
    assert old["valid_to"] == pd.Timestamp("2024-06-01")
    assert new["valid_from"] == pd.Timestamp("2024-06-01")
    assert new["valid_to"] == SCD2_FAR_FUTURE

    # tt2: unchanged, carried forward as the original current row.
    tt2 = snap[snap["title_id"] == "tt2"]
    assert len(tt2) == 1
    assert tt2.iloc[0]["valid_from"] == pd.Timestamp("2024-01-01")
    assert bool(tt2.iloc[0]["is_current"]) is True

    # tt3: new current row from the second run.
    tt3 = snap[snap["title_id"] == "tt3"]
    assert len(tt3) == 1
    assert tt3.iloc[0]["valid_from"] == pd.Timestamp("2024-06-01")
    assert bool(tt3.iloc[0]["is_current"]) is True

    # Exactly one current row per key.
    current_rows = snap[snap["is_current"]]
    assert current_rows["title_id"].nunique() == len(current_rows) == 3


def test_scd2_no_changes_is_stable() -> None:
    """Re-running with identical source data leaves the snapshot unchanged."""
    prev = _run1()
    current = pd.DataFrame(
        {"title_id": ["tt1", "tt2"], "title_type": ["movie", "short"]}
    )
    snap = scd2_snapshot(
        current, prev, key="title_id", cols=["title_type"], valid_from="2024-06-01"
    )
    assert len(snap) == 2
    assert snap["is_current"].all()
    # valid_from stayed at the original run, not the new one.
    assert (snap["valid_from"] == pd.Timestamp("2024-01-01")).all()


def test_scd2_deleted_key_keeps_history() -> None:
    """A key dropped from the source keeps its last version (no hard delete)."""
    prev = _run1()
    current = pd.DataFrame({"title_id": ["tt1"], "title_type": ["movie"]})
    snap = scd2_snapshot(
        current, prev, key="title_id", cols=["title_type"], valid_from="2024-06-01"
    )
    tt2 = snap[snap["title_id"] == "tt2"]
    assert len(tt2) == 1
    assert bool(tt2.iloc[0]["is_current"]) is True  # untouched


def test_scd2_null_attribute_unchanged_is_not_a_new_version() -> None:
    """NULL -> NULL counts as unchanged, so no spurious new version is opened."""
    prev = scd2_snapshot(
        pd.DataFrame({"k": ["a"], "v": [None]}),
        None,
        key="k",
        cols=["v"],
        valid_from="2024-01-01",
    )
    snap = scd2_snapshot(
        pd.DataFrame({"k": ["a"], "v": [None]}),
        prev,
        key="k",
        cols=["v"],
        valid_from="2024-06-01",
    )
    assert len(snap) == 1
    assert snap.iloc[0]["valid_from"] == pd.Timestamp("2024-01-01")


def test_scd2_empty_first_run() -> None:
    snap = scd2_snapshot(
        pd.DataFrame({"k": [], "v": []}),
        None,
        key="k",
        cols=["v"],
        valid_from="2024-01-01",
    )
    assert len(snap) == 0
    assert list(snap.columns) == ["k", "v", "valid_from", "valid_to", "is_current"]


def test_scd2_duplicate_key_raises() -> None:
    with pytest.raises(ValueError):
        scd2_snapshot(
            pd.DataFrame({"k": ["a", "a"], "v": [1, 2]}),
            None,
            key="k",
            cols=["v"],
            valid_from="2024-01-01",
        )


def test_scd2_empty_cols_raises() -> None:
    with pytest.raises(ValueError):
        scd2_snapshot(
            pd.DataFrame({"k": ["a"], "v": [1]}),
            None,
            key="k",
            cols=[],
            valid_from="2024-01-01",
        )
