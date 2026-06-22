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

* :func:`scd2_snapshot` — a Slowly-Changing-Dimension Type-2 snapshot step: given
  the current source rows and the previous versioned snapshot, it expires changed
  rows and appends new versions, the way ``dbt snapshot`` maintains history with
  ``valid_from`` / ``valid_to`` / ``is_current``.

All three are pure pandas/numpy/stdlib so they are importable and covered by
known-answer tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, timedelta

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


# Far-future sentinel for an open (still-current) SCD2 row's valid_to. The classic
# warehouse convention; a single NULL-free upper bound makes range joins simple.
SCD2_FAR_FUTURE = pd.Timestamp("9999-12-31")

_SCD2_META = ["valid_from", "valid_to", "is_current"]


def scd2_snapshot(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    key: str,
    cols: Sequence[str],
    valid_from: str | date | datetime,
) -> pd.DataFrame:
    """Advance a Slowly-Changing-Dimension Type-2 snapshot by one run.

    Compares the ``current`` source rows against the ``previous`` versioned
    snapshot and returns the new snapshot, the way ``dbt snapshot`` maintains
    history. Every output row carries three metadata columns:

    * ``valid_from`` — when this version became effective.
    * ``valid_to``   — when it was superseded; :data:`SCD2_FAR_FUTURE` while open.
    * ``is_current`` — bool, True for the live version of each key.

    Semantics, per key (identified by ``key``, with ``cols`` the tracked
    attributes):

    * **Unchanged** — the previous current row's tracked ``cols`` equal the
      source: it is carried forward untouched (same ``valid_from``).
    * **Changed**   — the previous current row is closed (``valid_to`` set to
      ``valid_from``, ``is_current`` False) and a new current version is appended
      with ``valid_from``.
    * **New**       — a key absent from the previous snapshot is inserted as a
      current row from ``valid_from``.
    * **Deleted**   — a key present before but absent from ``current`` keeps its
      history; its last version stays as it was (this step does not tombstone
      deletes, matching dbt snapshot's default ``invalidate_hard_deletes=false``).

    Already-expired history rows (``is_current == False`` in ``previous``) are
    always carried through unchanged.

    Parameters
    ----------
    current:
        The latest source rows. Must contain ``key`` and every column in
        ``cols``; one row per key.
    previous:
        The prior snapshot (output of an earlier call), or ``None`` / empty for
        the first run, in which case every current row is inserted new.
    key:
        Natural-key column identifying an entity across versions.
    cols:
        Tracked attribute columns; a change in any of them opens a new version.
    valid_from:
        Effective timestamp for this run's inserts and expirations.

    Returns
    -------
    pandas.DataFrame
        ``[key] + list(cols) + ["valid_from", "valid_to", "is_current"]``,
        ordered by ``key`` then ``valid_from``, with a fresh 0-based index.

    Raises
    ------
    ValueError
        If ``cols`` is empty, a required column is missing, or ``current`` has a
        duplicate ``key``.
    """
    if len(cols) == 0:
        raise ValueError("scd2_snapshot requires at least one tracked column.")
    track = list(cols)
    needed = [key, *track]
    missing = [c for c in needed if c not in current.columns]
    if missing:
        raise ValueError(
            f"Columns {missing!r} not in current columns {list(current.columns)!r}."
        )
    if current[key].duplicated().any():
        raise ValueError("current has duplicate key values; expected one row per key.")

    vf = pd.Timestamp(valid_from)
    out_cols = [key, *track, *_SCD2_META]

    # First run (or no prior history): every current row is a fresh current version.
    if previous is None or len(previous) == 0:
        fresh = current[needed].copy()
        fresh["valid_from"] = vf
        fresh["valid_to"] = SCD2_FAR_FUTURE
        fresh["is_current"] = True
        return (
            fresh[out_cols]
            .sort_values([key, "valid_from"])
            .reset_index(drop=True)
        )

    prev_missing = [c for c in [*needed, *_SCD2_META] if c not in previous.columns]
    if prev_missing:
        raise ValueError(
            f"Columns {prev_missing!r} not in previous columns "
            f"{list(previous.columns)!r}."
        )

    # Closed history rows pass straight through.
    history = previous[~previous["is_current"].astype(bool)].copy()
    prev_current = previous[previous["is_current"].astype(bool)].copy()

    cur_idx = current.set_index(key)
    prev_idx = prev_current.set_index(key)

    kept_current: list[pd.DataFrame] = []  # carried-forward unchanged current rows
    closed: list[pd.DataFrame] = []  # expired prior versions (changed keys)
    new_versions: list[dict] = []  # new current versions (changed + new keys)

    for k in cur_idx.index:
        src = cur_idx.loc[k]
        if k in prev_idx.index:
            prior = prev_idx.loc[k]
            same = all(_values_equal(prior[c], src[c]) for c in track)
            prior_row = prev_current[prev_current[key] == k]
            if same:
                kept_current.append(prior_row)
            else:
                expired = prior_row.copy()
                expired["valid_to"] = vf
                expired["is_current"] = False
                closed.append(expired)
                new_versions.append(_scd2_row(key, k, src, track, vf))
        else:
            new_versions.append(_scd2_row(key, k, src, track, vf))

    # Keys present before but gone from current keep their current row as-is.
    gone = prev_idx.index.difference(cur_idx.index)
    for k in gone:
        kept_current.append(prev_current[prev_current[key] == k])

    frames = [history, *closed, *kept_current]
    parts = [f[out_cols] for f in frames if len(f) > 0]
    if new_versions:
        parts.append(pd.DataFrame(new_versions, columns=out_cols))

    if not parts:
        return pd.DataFrame(columns=out_cols)

    result = pd.concat(parts, ignore_index=True)
    result["valid_from"] = pd.to_datetime(result["valid_from"])
    result["valid_to"] = pd.to_datetime(result["valid_to"])
    result["is_current"] = result["is_current"].astype(bool)
    return result.sort_values([key, "valid_from"]).reset_index(drop=True)


def _scd2_row(
    key: str,
    key_val: object,
    src: pd.Series,
    track: Sequence[str],
    vf: pd.Timestamp,
) -> dict:
    """Build one fresh current SCD2 row as a dict keyed by output column name."""
    row: dict = {key: key_val}
    for c in track:
        row[c] = src[c]
    row["valid_from"] = vf
    row["valid_to"] = SCD2_FAR_FUTURE
    row["is_current"] = True
    return row


def _values_equal(a: object, b: object) -> bool:
    """Equality treating two nulls as equal (so NULL -> NULL is 'unchanged')."""
    a_na = pd.isna(a)
    b_na = pd.isna(b)
    if a_na or b_na:
        return bool(a_na and b_na)
    return a == b
