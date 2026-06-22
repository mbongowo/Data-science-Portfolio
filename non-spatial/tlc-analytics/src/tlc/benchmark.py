"""Engine-agnostic benchmark harness (pure stdlib + pandas).

The bake-off times the *same* aggregation workload run through different
engines and ranks them. This module holds the parts that are pure and testable:
a wall-clock timer, a result container, and a ranking summary. The actual
engine execution (DuckDB SQL, a Spark job, a warehouse query) lives in
``engines.py`` and is imported lazily there — it is never needed to run, or to
test, the harness itself. The pandas reference path is fully exercised here.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd

T = TypeVar("T")


def duckdb_available() -> bool:
    """Return ``True`` iff ``duckdb`` can be imported, without importing it.

    Uses :func:`importlib.util.find_spec` so a probe never pulls the (heavy)
    engine module into the process. The bake-off uses this to decide between a
    real run and a clean "engine unavailable" skip.
    """
    return importlib.util.find_spec("duckdb") is not None


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


#: SQL for the hourly-demand mart, run against an in-memory view named ``trips``.
#: This mirrors :func:`tlc.marts.hourly_demand` on an already-cleaned frame.
HOURLY_DEMAND_SQL = (
    "SELECT EXTRACT(hour FROM pickup_datetime) AS hour, "
    "COUNT(*) AS trips "
    "FROM trips GROUP BY 1 ORDER BY 1"
)


def run_duckdb_query(df: pd.DataFrame, sql: str = HOURLY_DEMAND_SQL) -> pd.DataFrame:
    """Execute ``sql`` against ``df`` registered as the view ``trips``, via DuckDB.

    This is the *real* in-process DuckDB path used by the bake-off on the demo
    data: the frame is registered as a view named ``trips`` and the SQL is run
    and collected back to pandas. ``duckdb`` is imported **lazily** here so this
    module imports without it; call :func:`duckdb_available` first to guard.

    Parameters
    ----------
    df:
        A cleaned trips frame (must contain the columns the SQL references;
        :data:`HOURLY_DEMAND_SQL` needs ``pickup_datetime``).
    sql:
        SQL referencing the registered view ``trips``. Defaults to
        :data:`HOURLY_DEMAND_SQL`.

    Returns
    -------
    pandas.DataFrame
        The query result, collected to pandas.
    """
    import duckdb  # lazy: engine-only dependency

    con = duckdb.connect()
    try:
        con.register("trips", df)
        return con.execute(sql).fetch_df()
    finally:
        con.close()


def bake_off(df: pd.DataFrame, query: str = "hourly_demand") -> pd.DataFrame:
    """Run a real pandas-vs-DuckDB comparison on ``df`` and rank the engines.

    The pandas reference path (:func:`tlc.marts.hourly_demand`) is always timed.
    When :func:`duckdb_available` is ``True`` the equivalent SQL is executed
    in-process via :func:`run_duckdb_query` and timed too, so the returned table
    is a **measured** two-engine ranking. When DuckDB is absent, a single
    ``duckdb`` row is appended with ``seconds`` = NaN and ``status`` =
    ``"engine unavailable"`` so the skip is explicit rather than silent.

    Parameters
    ----------
    df:
        A cleaned trips frame (needs ``pickup_datetime`` for the default query).
    query:
        Label recorded in the result rows.

    Returns
    -------
    pandas.DataFrame
        The :func:`summarize` ranking with an extra ``status`` column
        (``"measured"`` or ``"engine unavailable"``). DuckDB rows that were not
        run sort last (NaN seconds) and are not ranked against real timings.
    """
    from tlc import marts  # local import avoids an import cycle at module load

    _, pandas_seconds = time_callable(lambda: marts.hourly_demand(df))
    results = [BenchmarkResult(engine="pandas", query=query, seconds=pandas_seconds)]
    statuses = {"pandas": "measured"}

    if duckdb_available():
        _, duckdb_seconds = time_callable(lambda: run_duckdb_query(df))
        results.append(
            BenchmarkResult(engine="duckdb", query=query, seconds=duckdb_seconds)
        )
        statuses["duckdb"] = "measured"
        ranking = summarize(results)
    else:
        ranking = summarize(results)
        unavailable = pd.DataFrame(
            {
                "engine": ["duckdb"],
                "query": [query],
                "seconds": [float("nan")],
                "peak_mb": [0.0],
                "rank": [0],
            }
        )
        ranking = pd.concat([ranking, unavailable], ignore_index=True)
        statuses["duckdb"] = "engine unavailable"

    ranking["status"] = ranking["engine"].map(statuses)
    return ranking
