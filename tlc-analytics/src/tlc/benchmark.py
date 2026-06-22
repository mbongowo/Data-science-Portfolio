"""Engine-agnostic benchmark harness (pure stdlib + pandas).

The bake-off times the *same* aggregation workload run through different
engines and ranks them. This module holds the parts that are pure and testable:
a wall-clock timer, a result container, and a ranking summary. The actual
engine execution (DuckDB SQL, a Spark job, a warehouse query) lives in
``engines.py`` and is imported lazily there — it is never needed to run, or to
test, the harness itself. The pandas reference path is fully exercised here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd

T = TypeVar("T")


def time_callable(fn: Callable[[], T]) -> tuple[T, float]:
    """Run ``fn`` once and time it with :func:`time.perf_counter`.

    Parameters
    ----------
    fn:
        A zero-argument callable (wrap arguments with ``functools.partial`` or a
        lambda). It is called exactly once.

    Returns
    -------
    tuple
        ``(result, seconds)`` where ``result`` is whatever ``fn`` returned and
        ``seconds`` is the elapsed wall-clock time, always non-negative.
    """
    start = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - start
    return result, seconds


@dataclass
class BenchmarkResult:
    """One timed run of one query on one engine.

    Attributes
    ----------
    engine:
        Engine label (e.g. ``"duckdb"``, ``"spark"``, ``"warehouse"``).
    query:
        Query / mart name (e.g. ``"hourly_demand"``).
    seconds:
        Wall-clock runtime in seconds.
    peak_mb:
        Peak resident memory in megabytes, if measured (else ``0.0``).
    """

    engine: str
    query: str
    seconds: float
    peak_mb: float = 0.0


def summarize(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Rank benchmark results fastest-first.

    Parameters
    ----------
    results:
        A list of :class:`BenchmarkResult`. May mix engines and queries.

    Returns
    -------
    pandas.DataFrame
        Columns ``engine``, ``query``, ``seconds``, ``peak_mb`` and ``rank``,
        sorted by ``seconds`` ascending. ``rank`` is a 1-based dense ranking on
        runtime (fastest = 1).

    Raises
    ------
    ValueError
        If ``results`` is empty.
    """
    if not results:
        raise ValueError("summarize: no benchmark results to rank.")

    df = pd.DataFrame(
        {
            "engine": [r.engine for r in results],
            "query": [r.query for r in results],
            "seconds": [r.seconds for r in results],
            "peak_mb": [r.peak_mb for r in results],
        }
    )
    df = df.sort_values("seconds", kind="stable").reset_index(drop=True)
    df["rank"] = df["seconds"].rank(method="dense").astype(int)
    return df
