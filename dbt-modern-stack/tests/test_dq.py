"""Known-answer tests for the pure-pandas data-quality core.

Each test plants exactly one kind of defect into a tiny DataFrame whose failure
count is obvious by hand, then checks the runner reports it. A green test proves
the generic-test semantics are correct, not merely that the code runs. No dbt /
duckdb dependency, so these always execute.

Worked fixtures (mirroring an IMDb staging table):

    titles
        tconst        title              titleType
        tt0000001     Carmencita         movie
        tt0000002     Le clown           short
        tt0000003     Pauvre Pierrot     short
        tt0000003     (DUPLICATE)        movie      <- planted duplicate tconst
        <null>        Orphan note        episode    <- planted null tconst

    ratings (child of titles on tconst)
        tconst        averageRating
        tt0000001     5.7
        tt0000002     6.1
        tt9999999     9.9                            <- planted orphan tconst
"""

from __future__ import annotations

import pandas as pd
import pytest

from dwh.dq import (
    TestSpec,
    run_suite,
    test_accepted_values,
    test_not_null,
    test_relationships,
    test_unique,
)


def _clean_titles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002", "tt0000003"],
            "title": ["Carmencita", "Le clown", "Pauvre Pierrot"],
            "titleType": ["movie", "short", "short"],
        }
    )


def test_not_null_detects_planted_null() -> None:
    """One null in the key column => exactly one not_null failure."""
    df = _clean_titles().copy()
    df.loc[len(df)] = [None, "Orphan note", "episode"]
    res = test_not_null(df, "tconst")
    assert res.test == "not_null"
    assert res.failures == 1
    assert res.passed is False


def test_not_null_passes_clean_column() -> None:
    res = test_not_null(_clean_titles(), "tconst")
    assert res.failures == 0
    assert res.passed is True


def test_unique_detects_planted_duplicate() -> None:
    """A repeated tconst => two rows in the duplicate group => 2 failures."""
    df = _clean_titles().copy()
    df.loc[len(df)] = ["tt0000003", "Pauvre Pierrot (dup)", "movie"]
    res = test_unique(df, "tconst")
    assert res.test == "unique"
    assert res.failures == 2
    assert res.passed is False
    assert "tt0000003" in res.detail


def test_unique_ignores_nulls() -> None:
    """dbt's unique does not flag NULLs; two NULLs alone still pass."""
    df = pd.DataFrame({"tconst": ["tt1", "tt2", None, None]})
    res = test_unique(df, "tconst")
    assert res.failures == 0
    assert res.passed is True


def test_accepted_values_flags_out_of_set() -> None:
    """One value outside the allowed set => one accepted_values failure."""
    df = _clean_titles().copy()
    df.loc[len(df)] = ["tt0000004", "Mystery", "hologram"]  # not an allowed type
    res = test_accepted_values(df, "titleType", ["movie", "short", "episode"])
    assert res.test == "accepted_values"
    assert res.failures == 1
    assert res.passed is False
    assert "hologram" in res.detail


def test_accepted_values_passes_when_all_in_set() -> None:
    res = test_accepted_values(
        _clean_titles(), "titleType", ["movie", "short", "episode"]
    )
    assert res.failures == 0
    assert res.passed is True


def test_relationships_flags_orphan_child_key() -> None:
    """A child key absent from the parent => one relationships failure."""
    parent = _clean_titles()
    child = pd.DataFrame(
        {
            "tconst": ["tt0000001", "tt0000002", "tt9999999"],
            "averageRating": [5.7, 6.1, 9.9],
        }
    )
    res = test_relationships(child, "tconst", parent, "tconst")
    assert res.test == "relationships"
    assert res.failures == 1
    assert res.passed is False
    assert "tt9999999" in res.detail


def test_relationships_passes_when_all_keys_match() -> None:
    parent = _clean_titles()
    child = pd.DataFrame({"tconst": ["tt0000001", "tt0000002"]})
    res = test_relationships(child, "tconst", parent, "tconst")
    assert res.failures == 0
    assert res.passed is True


def test_run_suite_summary_shape_and_contents() -> None:
    """run_suite returns one row per spec with the documented columns."""
    titles = _clean_titles().copy()
    titles.loc[len(titles)] = ["tt0000003", "dup", "movie"]  # plant a duplicate
    ratings = pd.DataFrame(
        {"tconst": ["tt0000001", "tt9999999"], "averageRating": [5.7, 9.9]}
    )

    suite = [
        TestSpec("not_null", titles, "tconst", table="stg_titles"),
        TestSpec("unique", titles, "tconst", table="stg_titles"),
        TestSpec(
            "accepted_values",
            titles,
            "titleType",
            table="stg_titles",
            values=["movie", "short", "episode"],
        ),
        TestSpec(
            "relationships",
            ratings,
            "tconst",
            table="fct_title_rating",
            parent=titles,
            parent_col="tconst",
        ),
    ]
    summary = run_suite(suite)

    assert list(summary.columns) == [
        "test",
        "table",
        "column",
        "passed",
        "failures",
        "detail",
    ]
    assert len(summary) == 4
    by_test = summary.set_index("test")
    assert by_test.loc["not_null", "passed"]  # no nulls planted
    assert not by_test.loc["unique", "passed"]  # duplicate planted
    assert by_test.loc["unique", "failures"] == 2
    assert by_test.loc["accepted_values", "passed"]  # all types valid
    assert not by_test.loc["relationships", "passed"]  # orphan planted
    assert by_test.loc["relationships", "failures"] == 1


def test_run_suite_empty_has_columns() -> None:
    summary = run_suite([])
    assert list(summary.columns) == [
        "test",
        "table",
        "column",
        "passed",
        "failures",
        "detail",
    ]
    assert len(summary) == 0


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        run_suite([TestSpec("nonexistent", _clean_titles(), "tconst")])


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        test_not_null(_clean_titles(), "no_such_column")
