"""Dimensional-modelling helpers used to build warehouse marts.

Two small, deterministic, dependency-light building blocks that the dbt marts
mirror in SQL:

* :func:`surrogate_key` — a stable hash key per row, the analytics-engineering
  convention for a warehouse-owned primary key that does not depend on a source
  system's natural key. dbt teams generate these with
  ``dbt_utils.generate_surrogate_key``; this is the same idea: hash the joined
  string of the chosen columns. Same inputs always give the same key, across
  runs and machines, so it is safe to use as a join key in incremental models.

* :func:`build_date_dim` — a classic date dimension (one row per calendar day),
  the spine almost every star schema joins facts against.

Both are pure pandas/numpy/stdlib so they are importable and covered by
known-answer tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

# Separator placed between column values before hashing. Using a delimiter that
# is very unlikely to appear in data avoids collisions like ("a", "bc") vs
# ("ab", "c").
_SEP = "|"


def surrogate_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Return a deterministic surrogate key per row.

    The key is the hex SHA-256 digest of the chosen columns' values joined by a
    delimiter. It is:

    * **Deterministic** — the same row values always hash to the same key, on
      any machine and across runs (unlike Python's salted ``hash()``).
    * **Distinct for distinct inputs** — two rows with different values in
      ``cols`` get different keys (barring a SHA-256 collision).

    Null/NaN values are rendered as the literal string ``"None"`` before
    hashing, so a missing value is stable but is treated as one possible value;
    enforce ``not_null`` on the source columns if a NULL key would be a bug.

    Parameters
    ----------
    df:
        Input frame.
    cols:
        Columns whose combined value identifies a row. Order matters.

    Returns
    -------
    pandas.Series
        String keys, indexed like ``df``.

    Raises
    ------
    ValueError
        If ``cols`` is empty or names a column not present in ``df``.
    """
    if len(cols) == 0:
        raise ValueError("surrogate_key requires at least one column.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing!r} not in DataFrame columns {list(df.columns)!r}."
        )

    def _hash_row(values: Sequence[object]) -> str:
        joined = _SEP.join("None" if pd.isna(v) else str(v) for v in values)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    sub = df[list(cols)]
    keys = [_hash_row(row) for row in sub.itertuples(index=False, name=None)]
    return pd.Series(keys, index=df.index, dtype="object", name="surrogate_key")


def build_date_dim(start: str | date, end: str | date) -> pd.DataFrame:
    """Build a date-dimension DataFrame, one row per day from ``start`` to ``end``.

    The range is **inclusive** of both endpoints, so the row count equals the
    number of days between them plus one.

    Columns:

    * ``date``        — ``datetime64`` calendar date
    * ``year``        — int
    * ``month``       — int, 1-12
    * ``day``         — int day of month, 1-31
    * ``dow``         — int day of week, Monday=0 .. Sunday=6 (pandas convention)
    * ``is_weekend``  — bool, True on Saturday/Sunday

    Parameters
    ----------
    start, end:
        Inclusive bounds, as ``date`` objects or ISO ``"YYYY-MM-DD"`` strings.

    Returns
    -------
    pandas.DataFrame
        The date dimension, ordered by date with a fresh 0-based index.

    Raises
    ------
    ValueError
        If ``end`` is before ``start``.
    """
    start_d = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    if end_d < start_d:
        raise ValueError(f"end ({end_d.date()}) is before start ({start_d.date()}).")

    dates = pd.date_range(start=start_d, end=end_d, freq="D")
    dow = dates.dayofweek
    return pd.DataFrame(
        {
            "date": dates,
            "year": dates.year.astype(int),
            "month": dates.month.astype(int),
            "day": dates.day.astype(int),
            "dow": dow.astype(int),
            "is_weekend": dow >= 5,
        }
    ).reset_index(drop=True)


def _days_inclusive(start: date, end: date) -> int:
    """Return the inclusive day count between two dates (helper / documentation)."""
    return (end - start + timedelta(days=1)).days
