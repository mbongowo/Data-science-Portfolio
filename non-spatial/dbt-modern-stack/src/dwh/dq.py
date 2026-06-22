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

Beyond dbt's four built-ins, this module also ships a handful of tests analytics
engineers usually reach for via ``dbt_utils`` or singular tests:

* ``accepted_range``  — fails on each row whose value is outside ``[lo, hi]``
                        (``dbt_utils.accepted_range``). Null values are ignored.
* ``not_null_where``  — ``not_null`` restricted to the rows matching a predicate
                        (``dbt_utils.not_null_where``); only flags nulls there.
* ``expression``      — a row-level boolean expression that must hold for every
                        row (``dbt_utils.expression_is_true``); fails each row
                        where it is False.
* ``freshness``       — relational source freshness: fails when the newest value
                        of a timestamp column is older than ``max_age`` relative
                        to ``now`` (mirrors ``dbt source freshness``).

Every test can run at a **severity** of ``"error"`` (default) or ``"warn"``.
A warning that fails does not fail the suite; :func:`run_suite` reports the two
separately, just as dbt's ``severity: warn`` config does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


def test_accepted_range(
    df: pd.DataFrame,
    column: str,
    lo: float | None = None,
    hi: float | None = None,
    *,
    inclusive: bool = True,
    table: str = "",
) -> TestResult:
    """Fail on each row outside ``[lo, hi]`` (``dbt_utils.accepted_range``).

    Either bound may be ``None`` to leave that side unbounded (a pure minimum or
    maximum check). With ``inclusive=True`` (the default) the bounds themselves
    pass; with ``inclusive=False`` a value equal to a bound fails. Null values
    are ignored, matching the dbt_utils macro (pair with ``not_null`` if a NULL
    should itself be a failure).
    """
    _require_column(df, column)
    if lo is None and hi is None:
        raise ValueError("accepted_range requires at least one of `lo`, `hi`.")
    col = df[column]
    present = col.notna()
    out = pd.Series(False, index=col.index)
    if lo is not None:
        out |= (col < lo) if inclusive else (col <= lo)
    if hi is not None:
        out |= (col > hi) if inclusive else (col >= hi)
    mask = present & out
    failures = int(mask.sum())
    if failures == 0:
        detail = ""
    else:
        bracket = "[]" if inclusive else "()"
        lo_s = "-inf" if lo is None else str(lo)
        hi_s = "+inf" if hi is None else str(hi)
        detail = f"{failures} value(s) outside {bracket[0]}{lo_s}, {hi_s}{bracket[1]}"
    return TestResult(
        test="accepted_range",
        table=table,
        column=column,
        passed=failures == 0,
        failures=failures,
        detail=detail,
    )


def test_not_null_where(
    df: pd.DataFrame,
    column: str,
    predicate: Callable[[pd.DataFrame], pd.Series],
    *,
    table: str = "",
) -> TestResult:
    """Fail on each null in ``column`` among rows matching ``predicate``.

    ``predicate`` takes the frame and returns a boolean Series (the rows the test
    applies to), mirroring ``dbt_utils.not_null_where``'s ``where:`` clause. Rows
    outside the predicate are never flagged, even if ``column`` is null there.
    """
    _require_column(df, column)
    where = _as_bool_mask(predicate(df), df.index, "not_null_where predicate")
    mask = where & df[column].isna()
    failures = int(mask.sum())
    return TestResult(
        test="not_null_where",
        table=table,
        column=column,
        passed=failures == 0,
        failures=failures,
        detail=(
            ""
            if failures == 0
            else f"{failures} null value(s) where predicate holds"
        ),
    )


def test_expression(
    df: pd.DataFrame,
    expr_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    column: str = "",
    table: str = "",
) -> TestResult:
    """Fail on each row where ``expr_fn`` is not True (``expression_is_true``).

    ``expr_fn`` takes the frame and returns a row-level boolean Series that must
    hold for every row. Any row where it is ``False`` (or null/NaN, which is not
    True) is a failure. This is the pandas analogue of dbt's
    ``dbt_utils.expression_is_true`` generic test, used for cross-column rules a
    single-column test cannot state (e.g. ``end_year >= start_year``).
    """
    raw = expr_fn(df)
    truth = _as_bool_mask(raw, df.index, "expression result")
    mask = ~truth
    failures = int(mask.sum())
    return TestResult(
        test="expression",
        table=table,
        column=column,
        passed=failures == 0,
        failures=failures,
        detail="" if failures == 0 else f"{failures} row(s) violate the expression",
    )


def test_freshness(
    df: pd.DataFrame,
    ts_col: str,
    max_age: timedelta,
    now: datetime,
    *,
    table: str = "",
) -> TestResult:
    """Fail (1) when the newest ``ts_col`` is older than ``max_age`` before ``now``.

    This mirrors ``dbt source freshness``: the source is stale when its most
    recent load timestamp lags ``now`` by more than ``max_age``. The result is a
    relational pass/fail, so ``failures`` is ``1`` when stale and ``0`` when
    fresh (not a per-row count). An empty frame, or one with only null
    timestamps, is treated as stale (there is no fresh evidence).
    """
    _require_column(df, ts_col)
    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
    now_ts = pd.Timestamp(now)
    if len(ts) == 0:
        return TestResult(
            test="freshness",
            table=table,
            column=ts_col,
            passed=False,
            failures=1,
            detail="no non-null timestamps; treated as stale",
        )
    newest = ts.max()
    age = now_ts - newest
    stale = age > max_age
    return TestResult(
        test="freshness",
        table=table,
        column=ts_col,
        passed=not stale,
        failures=1 if stale else 0,
        detail="" if not stale else f"newest row is {age} old (> {max_age})",
    )


@dataclass
class TestSpec:
    """A single test to run in a suite.

    Mirrors a line in a dbt ``_*.yml`` test block. ``kind`` is one of
    ``not_null``, ``unique``, ``accepted_values``, ``relationships``,
    ``accepted_range``, ``not_null_where``, ``expression``, ``freshness``.

    Set the fields the kind needs:

    * ``accepted_values`` — ``values``
    * ``relationships``   — ``parent`` (a DataFrame) and ``parent_col``
    * ``accepted_range``  — ``lo`` and/or ``hi`` (optionally ``inclusive`` in
      ``extra``)
    * ``not_null_where``  — ``predicate``
    * ``expression``      — ``expr_fn`` (``column`` is optional, for labelling)
    * ``freshness``       — ``ts_col`` (defaults to ``column``), ``max_age``,
      ``now``

    ``severity`` is ``"error"`` (default) or ``"warn"``. A failing ``warn`` test
    does not fail the suite; :func:`run_suite` counts it separately.
    """

    kind: str
    df: pd.DataFrame
    column: str = ""
    table: str = ""
    values: Sequence[Any] | None = None
    parent: pd.DataFrame | None = None
    parent_col: str | None = None
    lo: float | None = None
    hi: float | None = None
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None
    expr_fn: Callable[[pd.DataFrame], pd.Series] | None = None
    ts_col: str | None = None
    max_age: timedelta | None = None
    now: datetime | None = None
    severity: str = "error"
    extra: dict[str, Any] = field(default_factory=dict)


_SUMMARY_COLUMNS = [
    "test",
    "table",
    "column",
    "severity",
    "passed",
    "failures",
    "detail",
]


def run_suite(suite: Iterable[TestSpec]) -> pd.DataFrame:
    """Run a list of :class:`TestSpec` and summarise like ``dbt test``.

    Returns a DataFrame with one row per test and the columns
    ``["test", "table", "column", "severity", "passed", "failures", "detail"]``.
    The frame is empty (but with those columns) when ``suite`` is empty, so
    callers can rely on the schema.

    Each spec carries a ``severity`` of ``"error"`` (default) or ``"warn"``.
    A failing ``warn`` test still reports ``passed == False`` on its own row, but
    :func:`suite_failed` (and the summary's notion of "did the suite fail") only
    counts failing ``error`` tests, matching dbt's ``severity: warn``.

    Raises
    ------
    ValueError
        If a spec has an unknown ``kind``, an invalid ``severity``, or is missing
        the fields that kind needs (e.g. ``accepted_values`` without ``values``).
    """
    rows: list[tuple[TestResult, str]] = []
    for spec in suite:
        if spec.severity not in ("error", "warn"):
            raise ValueError(
                f"severity must be 'error' or 'warn', got {spec.severity!r}."
            )
        rows.append((_run_one(spec), spec.severity))

    if not rows:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    return pd.DataFrame(
        [
            {
                "test": r.test,
                "table": r.table,
                "column": r.column,
                "severity": severity,
                "passed": r.passed,
                "failures": r.failures,
                "detail": r.detail,
            }
            for r, severity in rows
        ],
        columns=_SUMMARY_COLUMNS,
    )


def suite_failed(summary: pd.DataFrame) -> bool:
    """True if any ``error``-severity test failed in a :func:`run_suite` summary.

    Failing ``warn`` tests are ignored here, mirroring dbt: a warning surfaces in
    the summary but does not make ``dbt build`` exit non-zero.
    """
    if len(summary) == 0:
        return False
    errors = summary[summary["severity"] == "error"]
    return bool((~errors["passed"]).any())


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
    if kind == "accepted_range":
        if spec.lo is None and spec.hi is None:
            raise ValueError("accepted_range spec requires `lo` and/or `hi`.")
        return test_accepted_range(
            spec.df,
            spec.column,
            spec.lo,
            spec.hi,
            inclusive=bool(spec.extra.get("inclusive", True)),
            table=spec.table,
        )
    if kind == "not_null_where":
        if spec.predicate is None:
            raise ValueError("not_null_where spec requires `predicate`.")
        return test_not_null_where(
            spec.df, spec.column, spec.predicate, table=spec.table
        )
    if kind == "expression":
        if spec.expr_fn is None:
            raise ValueError("expression spec requires `expr_fn`.")
        return test_expression(
            spec.df, spec.expr_fn, column=spec.column, table=spec.table
        )
    if kind == "freshness":
        if spec.max_age is None or spec.now is None:
            raise ValueError("freshness spec requires `max_age` and `now`.")
        ts_col = spec.ts_col or spec.column
        if not ts_col:
            raise ValueError("freshness spec requires `ts_col` (or `column`).")
        return test_freshness(
            spec.df, ts_col, spec.max_age, spec.now, table=spec.table
        )
    raise ValueError(f"Unknown test kind: {kind!r}")


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(
            f"Column {column!r} not in DataFrame columns {list(df.columns)!r}."
        )


def _as_bool_mask(series: Any, index: pd.Index, what: str) -> pd.Series:
    """Coerce a predicate/expression result to a clean boolean Series.

    Aligns to ``index`` and treats null/NaN as ``False`` (a missing truth value
    is not a satisfied condition). Raises if the result is not Series-shaped over
    the frame's rows.
    """
    if not isinstance(series, pd.Series):
        raise ValueError(f"{what} must return a pandas Series, got {type(series)!r}.")
    if not series.index.equals(index):
        series = series.reindex(index)
    return series.fillna(False).astype(bool)


# The public callables and dataclasses are named after dbt's generic tests, so
# their names begin with ``test``/``Test``. Tell pytest not to collect them as
# test items (they are library code, exercised by tests/test_dq.py).
TestResult.__test__ = False
TestSpec.__test__ = False
test_not_null.__test__ = False  # type: ignore[attr-defined]
test_unique.__test__ = False  # type: ignore[attr-defined]
test_accepted_values.__test__ = False  # type: ignore[attr-defined]
test_relationships.__test__ = False  # type: ignore[attr-defined]
test_accepted_range.__test__ = False  # type: ignore[attr-defined]
test_not_null_where.__test__ = False  # type: ignore[attr-defined]
test_expression.__test__ = False  # type: ignore[attr-defined]
test_freshness.__test__ = False  # type: ignore[attr-defined]
