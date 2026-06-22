"""A pure-pandas data-quality runner that mirrors dbt's generic tests.

dbt ships four generic tests that analytics engineers reach for constantly:
``not_null``, ``unique``, ``accepted_values``, and ``relationships``. Inside the
warehouse those compile to ``SELECT`` statements that return the *failing* rows;
a test passes when that query returns zero rows.

This module reimplements the same four semantics over pandas DataFrames, with no
warehouse and no dbt dependency, so the logic is importable and testable on its
own. Each test returns a small :class:`TestResult` (``passed``: bool,
``failures``: int, plus context), and :func:`run_suite` collects a list of test
specs into a summary DataFrame that reads like ``dbt test`` output.

The point of the pure-pandas core is twofold:

1. It is the part of the stack that *must* keep working, so it is covered by
   hand-derived known-answer tests with no heavy dependencies.
2. It documents exactly what each generic test means, including the edge cases
   (NULLs in a unique column, NULLs in a relationships child key) where dbt's
   behaviour surprises people.

Failure semantics match dbt:

* ``not_null``        — fails on each row whose value is null/NaN.
* ``unique``          — fails on each row that shares its value with another row.
                        Null values are ignored (dbt's ``unique`` does not flag
                        NULLs; pair it with ``not_null`` if you need both).
* ``accepted_values`` — fails on each row whose value is outside the allowed set.
                        Null values are ignored, matching dbt.
* ``relationships``   — fails on each child row whose key has no match in the
                        parent column (an orphan / broken foreign key). Null
                        child keys are ignored, matching dbt.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TestResult:
    """Outcome of one data-quality test.

    Note for pytest: the names in this module start with ``test``/``Test`` to
    mirror dbt's vocabulary, not pytest's. ``__test__ = False`` is set on each
    so pytest does not try to collect them as test cases.

    Attributes
    ----------
    test:
        Name of the generic test (``not_null``, ``unique``, ...).
    table:
        Label for the table the test ran against (free text).
    column:
        Column the test ran against.
    passed:
        True when ``failures == 0``.
    failures:
        Number of failing rows (the row count dbt's test query would return).
    detail:
        Free-text context, e.g. the offending values or the parent column.
    """

    test: str
    table: str
    column: str
    passed: bool
    failures: int
    detail: str = ""


def test_not_null(df: pd.DataFrame, column: str, *, table: str = "") -> TestResult:
    """Fail on each row where ``column`` is null/NaN (dbt ``not_null``)."""
    _require_column(df, column)
    mask = df[column].isna()
    failures = int(mask.sum())
    return TestResult(
        test="not_null",
        table=table,
        column=column,
        passed=failures == 0,
        failures=failures,
        detail="" if failures == 0 else f"{failures} null value(s)",
    )


def test_unique(df: pd.DataFrame, column: str, *, table: str = "") -> TestResult:
    """Fail on each row that shares its value with another (dbt ``unique``).

    Null values are ignored, matching dbt: a column of all-distinct non-null
    values plus some NULLs still passes ``unique``. The failure count is the
    number of rows involved in a duplicate group (so two rows sharing one value
    count as two failures, as dbt's ``SELECT ... GROUP BY ... HAVING count > 1``
    expanded back to rows would report).
    """
    _require_column(df, column)
    non_null = df[column].dropna()
    dup_mask = non_null.duplicated(keep=False)
    failures = int(dup_mask.sum())
    if failures == 0:
        detail = ""
    else:
        dup_values = sorted(map(str, non_null[dup_mask].unique()))
        detail = f"duplicated value(s): {', '.join(dup_values)}"
    return TestResult(
        test="unique",
        table=table,
        column=column,
        passed=failures == 0,
        failures=failures,
        detail=detail,
    )


def test_accepted_values(
    df: pd.DataFrame,
    column: str,
    values: Iterable[Any],
    *,
    table: str = "",
) -> TestResult:
    """Fail on each row whose value is outside ``values`` (dbt ``accepted_values``).

    Null values are ignored, matching dbt (combine with ``not_null`` if NULLs
    should themselves be a failure).
    """
    _require_column(df, column)
    allowed = set(values)
    col = df[column]
    mask = col.notna() & ~col.isin(allowed)
    failures = int(mask.sum())
    if failures == 0:
        detail = ""
    else:
        bad = sorted(map(str, col[mask].unique()))
        detail = f"unexpected value(s): {', '.join(bad)}"
    return TestResult(
        test="accepted_values",
        table=table,
        column=column,
        passed=failures == 0,
        failures=failures,
        detail=detail,
    )


def test_relationships(
    child_df: pd.DataFrame,
    child_col: str,
    parent_df: pd.DataFrame,
    parent_col: str,
    *,
    table: str = "",
) -> TestResult:
    """Fail on each orphan child key (dbt ``relationships`` / referential integrity).

    A child row fails when its key has no matching value in the parent column.
    Null child keys are ignored, matching dbt: an unenforced foreign key is only
    broken when it points at a value the parent does not have.
    """
    _require_column(child_df, child_col)
    _require_column(parent_df, parent_col)
    parent_keys = set(parent_df[parent_col].dropna())
    child = child_df[child_col]
    mask = child.notna() & ~child.isin(parent_keys)
    failures = int(mask.sum())
    if failures == 0:
        detail = ""
    else:
        orphans = sorted(map(str, child[mask].unique()))
        detail = f"orphan key(s): {', '.join(orphans)}"
    return TestResult(
        test="relationships",
        table=table,
        column=child_col,
        passed=failures == 0,
        failures=failures,
        detail=detail,
    )


@dataclass
class TestSpec:
    """A single test to run in a suite.

    Mirrors a line in a dbt ``_*.yml`` test block. ``kind`` is one of
    ``not_null``, ``unique``, ``accepted_values``, ``relationships``.

    For ``accepted_values`` set ``values``. For ``relationships`` set
    ``parent`` (a DataFrame) and ``parent_col``.
    """

    kind: str
    df: pd.DataFrame
    column: str
    table: str = ""
    values: Sequence[Any] | None = None
    parent: pd.DataFrame | None = None
    parent_col: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_SUMMARY_COLUMNS = ["test", "table", "column", "passed", "failures", "detail"]


def run_suite(suite: Iterable[TestSpec]) -> pd.DataFrame:
    """Run a list of :class:`TestSpec` and summarise like ``dbt test``.

    Returns a DataFrame with one row per test and the columns
    ``["test", "table", "column", "passed", "failures", "detail"]``. The frame
    is empty (but with those columns) when ``suite`` is empty, so callers can
    rely on the schema.

    Raises
    ------
    ValueError
        If a spec has an unknown ``kind`` or is missing the fields that kind
        needs (e.g. ``accepted_values`` without ``values``).
    """
    rows: list[TestResult] = []
    for spec in suite:
        rows.append(_run_one(spec))

    if not rows:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    return pd.DataFrame(
        [
            {
                "test": r.test,
                "table": r.table,
                "column": r.column,
                "passed": r.passed,
                "failures": r.failures,
                "detail": r.detail,
            }
            for r in rows
        ],
        columns=_SUMMARY_COLUMNS,
    )


def _run_one(spec: TestSpec) -> TestResult:
    kind = spec.kind
    if kind == "not_null":
        return test_not_null(spec.df, spec.column, table=spec.table)
    if kind == "unique":
        return test_unique(spec.df, spec.column, table=spec.table)
    if kind == "accepted_values":
        if spec.values is None:
            raise ValueError("accepted_values spec requires `values`.")
        return test_accepted_values(spec.df, spec.column, spec.values, table=spec.table)
    if kind == "relationships":
        if spec.parent is None or spec.parent_col is None:
            raise ValueError("relationships spec requires `parent` and `parent_col`.")
        return test_relationships(
            spec.df, spec.column, spec.parent, spec.parent_col, table=spec.table
        )
    raise ValueError(f"Unknown test kind: {kind!r}")


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(
            f"Column {column!r} not in DataFrame columns {list(df.columns)!r}."
        )


# The public callables and dataclasses are named after dbt's generic tests, so
# their names begin with ``test``/``Test``. Tell pytest not to collect them as
# test items (they are library code, exercised by tests/test_dq.py).
TestResult.__test__ = False
TestSpec.__test__ = False
test_not_null.__test__ = False  # type: ignore[attr-defined]
test_unique.__test__ = False  # type: ignore[attr-defined]
test_accepted_values.__test__ = False  # type: ignore[attr-defined]
test_relationships.__test__ = False  # type: ignore[attr-defined]
