"""Partition-path logic for the Hive-partitioned TLC Parquet lake.

The engines read a ``year=/month=`` lake (see ``data/README.md``). This module
is the **pure path logic** for that layout: given a root and the years/months in
scope, :func:`iter_partitions` yields the expected partition directory paths in a
deterministic order. It performs **no I/O** — it does not stat or glob the
filesystem — so it is trivially testable and is the single place that knows the
``year=YYYY/month=MM`` naming convention.

The engine runners and the CLI can use these paths to build a read glob or to
prune partitions; nothing here depends on pandas or an engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath

#: The Hive partition keys, outermost first.
PARTITION_KEYS = ("year", "month")


def partition_relpath(year: int, month: int) -> str:
    """Return the relative ``year=YYYY/month=MM`` path for one partition.

    ``month`` is zero-padded to two digits (Hive convention); ``year`` is not
    padded. The separator is always ``/`` so the value is stable across
    platforms and usable directly in a DuckDB / Spark read glob.

    Raises
    ------
    ValueError
        If ``month`` is not in ``1..12``.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return f"year={year}/month={month:02d}"


def iter_partitions(
    root: str,
    years: Iterable[int],
    months: Iterable[int],
) -> Iterator[str]:
    """Yield expected partition paths under ``root`` for the year/month grid.

    The cross-product of ``years`` x ``months`` is walked **in the given order**
    (years outer, months inner) and one path per cell is yielded as a string,
    joined under ``root`` with forward slashes (``{root}/year=Y/month=MM``).

    This is pure path arithmetic: no path is checked for existence, so the
    caller gets every *expected* partition whether or not the file has been
    downloaded yet. Deduplication is **not** performed — duplicate inputs yield
    duplicate paths — so the output length is exactly
    ``len(list(years)) * len(list(months))``.

    Parameters
    ----------
    root:
        Lake root, e.g. ``"data/raw/yellow"``. Trailing slashes are tolerated.
    years:
        Iterable of integer years (e.g. ``range(2021, 2024)``).
    months:
        Iterable of integer months in ``1..12``.

    Yields
    ------
    str
        ``"{root}/year={year}/month={month:02d}"`` for each cell, in order.
    """
    base = PurePosixPath(str(root).rstrip("/") or ".")
    months = list(months)
    for year in years:
        for month in months:
            yield (base / partition_relpath(year, month)).as_posix()
