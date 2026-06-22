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

from datetime import datetime, timedelta

import pandas as pd
import pytest

from dwh.dq import (
    TestSpec,
    run_suite,
    suite_failed,
    test_accepted_range,
    test_accepted_values,
    test_expression,
    test_freshness,
    test_not_null,
    test_not_null_where,
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
        "severity",
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
        "severity",
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


# --------------------------------------------------------------------------- #
# accepted_range
# --------------------------------------------------------------------------- #
#
# Known-answer fixture (averageRating, allowed [1.0, 10.0] inclusive):
#
#     5.7  ok
#     0.0  too low   <- failure
#    10.0  ok (boundary, inclusive)
#    11.2  too high  <- failure
#
# Worked by hand: 2 failures.


def _ratings_for_range() -> pd.DataFrame:
    return pd.DataFrame({"averageRating": [5.7, 0.0, 10.0, 11.2]})


def test_accepted_range_flags_out_of_bounds() -> None:
    res = test_accepted_range(_ratings_for_range(), "averageRating", 1.0, 10.0)
    assert res.test == "accepted_range"
    assert res.failures == 2  # 0.0 (too low) and 11.2 (too high)
    assert res.passed is False


def test_accepted_range_inclusive_boundary_passes() -> None:
    """With inclusive bounds, a value equal to hi (10.0) passes."""
    df = pd.DataFrame({"x": [10.0]})
    assert test_accepted_range(df, "x", 1.0, 10.0).failures == 0


def test_accepted_range_exclusive_boundary_fails() -> None:
    """With inclusive=False, the boundary value 10.0 now fails."""
    df = pd.DataFrame({"x": [10.0]})
    res = test_accepted_range(df, "x", 1.0, 10.0, inclusive=False)
    assert res.failures == 1


def test_accepted_range_one_sided_minimum() -> None:
    """Only a lower bound: 0 and -3 are below 1, so 2 failures."""
    df = pd.DataFrame({"x": [0, -3, 5, 100]})
    res = test_accepted_range(df, "x", lo=1.0, hi=None)
    assert res.failures == 2


def test_accepted_range_ignores_nulls() -> None:
    df = pd.DataFrame({"x": [None, 5.0, None]})
    assert test_accepted_range(df, "x", 1.0, 10.0).failures == 0


def test_accepted_range_no_bounds_raises() -> None:
    with pytest.raises(ValueError):
        test_accepted_range(_ratings_for_range(), "averageRating")


# --------------------------------------------------------------------------- #
# not_null_where
# --------------------------------------------------------------------------- #
#
# Known-answer fixture: num_votes must be present only for released titles.
#
#     status     num_votes
#     released   1200        ok
#     released   <null>      failure (released but no votes)
#     upcoming   <null>      ignored (predicate excludes it)
#     released   3           ok
#
# Worked by hand: predicate (status == "released") matches 3 rows; one is null
# => 1 failure.


def _votes_with_status() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status": ["released", "released", "upcoming", "released"],
            "num_votes": [1200, None, None, 3],
        }
    )


def test_not_null_where_flags_only_predicate_rows() -> None:
    df = _votes_with_status()
    res = test_not_null_where(
        df, "num_votes", lambda d: d["status"] == "released"
    )
    assert res.test == "not_null_where"
    assert res.failures == 1  # the released row with a null vote count
    assert res.passed is False


def test_not_null_where_predicate_matches_nothing() -> None:
    """If the predicate excludes every row, a null column still passes."""
    df = _votes_with_status()
    res = test_not_null_where(df, "num_votes", lambda d: d["status"] == "cancelled")
    assert res.failures == 0
    assert res.passed is True


def test_not_null_where_predicate_all_rows_equals_not_null() -> None:
    """An all-True predicate reduces to plain not_null: 1 null => 1 failure."""
    df = _votes_with_status()
    res = test_not_null_where(df, "num_votes", lambda d: d["status"].notna())
    # released null + upcoming null = 2 nulls overall.
    assert res.failures == 2


# --------------------------------------------------------------------------- #
# expression (row-level boolean rule)
# --------------------------------------------------------------------------- #
#
# Known-answer fixture: end_year must be >= start_year.
#
#     start_year  end_year
#     2000        2005       ok
#     2010        2009       failure (ends before it starts)
#     1999        1999       ok (equal)
#
# Worked by hand: 1 failure.


def _year_spans() -> pd.DataFrame:
    return pd.DataFrame(
        {"start_year": [2000, 2010, 1999], "end_year": [2005, 2009, 1999]}
    )


def test_expression_flags_violating_rows() -> None:
    res = test_expression(
        _year_spans(), lambda d: d["end_year"] >= d["start_year"], column="end_year"
    )
    assert res.test == "expression"
    assert res.failures == 1
    assert res.passed is False


def test_expression_all_true_passes() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    res = test_expression(df, lambda d: d["a"] > 0)
    assert res.failures == 0
    assert res.passed is True


def test_expression_null_result_counts_as_failure() -> None:
    """A null truth value is not True, so it is a failure (1 here)."""
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    res = test_expression(df, lambda d: d["a"] > 0)
    assert res.failures == 1


# --------------------------------------------------------------------------- #
# freshness (relational source freshness)
# --------------------------------------------------------------------------- #
#
# Known-answer fixture: now = 2026-01-10 12:00; max_age = 24h.
# Newest load timestamp 2026-01-10 09:00 is 3h old => fresh (0 failures).
# Newest load timestamp 2026-01-08 09:00 is ~51h old => stale (1 failure).


_NOW = datetime(2026, 1, 10, 12, 0, 0)


def test_freshness_fresh_when_recent() -> None:
    df = pd.DataFrame(
        {"_loaded_at": ["2026-01-09 09:00", "2026-01-10 09:00"]}
    )
    res = test_freshness(df, "_loaded_at", timedelta(hours=24), _NOW)
    assert res.test == "freshness"
    assert res.failures == 0
    assert res.passed is True


def test_freshness_stale_when_old() -> None:
    df = pd.DataFrame(
        {"_loaded_at": ["2026-01-07 09:00", "2026-01-08 09:00"]}
    )
    res = test_freshness(df, "_loaded_at", timedelta(hours=24), _NOW)
    assert res.failures == 1  # relational: stale is a single failure
    assert res.passed is False


def test_freshness_empty_is_stale() -> None:
    df = pd.DataFrame({"_loaded_at": pd.Series([], dtype="object")})
    res = test_freshness(df, "_loaded_at", timedelta(hours=24), _NOW)
    assert res.passed is False
    assert res.failures == 1


def test_freshness_all_null_is_stale() -> None:
    df = pd.DataFrame({"_loaded_at": [None, None]})
    res = test_freshness(df, "_loaded_at", timedelta(hours=24), _NOW)
    assert res.passed is False


# --------------------------------------------------------------------------- #
# severity levels in run_suite
# --------------------------------------------------------------------------- #


def test_run_suite_warn_does_not_fail_suite() -> None:
    """A failing warn-severity test reports passed=False but does not fail the suite."""
    df = pd.DataFrame({"x": [None, 1.0]})  # one null
    suite = [
        TestSpec("not_null", df, "x", table="t", severity="warn"),
    ]
    summary = run_suite(suite)
    assert "severity" in summary.columns
    assert summary.loc[0, "severity"] == "warn"
    assert summary.loc[0, "passed"] is False or not bool(summary.loc[0, "passed"])
    assert suite_failed(summary) is False  # warn does not fail the suite


def test_run_suite_error_fails_suite() -> None:
    df = pd.DataFrame({"x": [None, 1.0]})
    summary = run_suite([TestSpec("not_null", df, "x", table="t")])  # default error
    assert summary.loc[0, "severity"] == "error"
    assert suite_failed(summary) is True


def test_run_suite_mixed_severity_only_error_fails() -> None:
    df = pd.DataFrame({"x": [None, 1.0]})
    suite = [
        TestSpec("not_null", df, "x", table="t", severity="warn"),
        TestSpec("unique", df, "x", table="t", severity="error"),  # passes
    ]
    summary = run_suite(suite)
    assert suite_failed(summary) is False  # only the warn failed


def test_run_suite_invalid_severity_raises() -> None:
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError):
        run_suite([TestSpec("not_null", df, "x", severity="critical")])


def test_run_suite_dispatches_new_kinds() -> None:
    """run_suite wires the new kinds end to end (range, expression, freshness)."""
    df = pd.DataFrame(
        {
            "rating": [5.0, 99.0],  # 99 is out of [1, 10]
            "start_year": [2000, 2010],
            "end_year": [2005, 2009],  # second row violates end >= start
            "_loaded_at": ["2026-01-08 09:00", "2026-01-08 09:00"],  # stale vs _NOW
        }
    )
    suite = [
        TestSpec("accepted_range", df, "rating", lo=1.0, hi=10.0, table="t"),
        TestSpec(
            "expression",
            df,
            column="end_year",
            table="t",
            expr_fn=lambda d: d["end_year"] >= d["start_year"],
        ),
        TestSpec(
            "freshness",
            df,
            table="t",
            ts_col="_loaded_at",
            max_age=timedelta(hours=24),
            now=_NOW,
        ),
        TestSpec(
            "not_null_where",
            df,
            "rating",
            table="t",
            predicate=lambda d: d["rating"] > 0,
        ),
    ]
    summary = run_suite(suite)
    by_test = summary.set_index("test")
    assert by_test.loc["accepted_range", "failures"] == 1
    assert by_test.loc["expression", "failures"] == 1
    assert by_test.loc["freshness", "failures"] == 1
    assert by_test.loc["not_null_where", "failures"] == 0


def test_run_suite_missing_kind_fields_raise() -> None:
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError):
        run_suite([TestSpec("accepted_range", df, "x")])  # no lo/hi
    with pytest.raises(ValueError):
        run_suite([TestSpec("not_null_where", df, "x")])  # no predicate
    with pytest.raises(ValueError):
        run_suite([TestSpec("expression", df, "x")])  # no expr_fn
    with pytest.raises(ValueError):
        run_suite([TestSpec("freshness", df, "x")])  # no max_age/now
