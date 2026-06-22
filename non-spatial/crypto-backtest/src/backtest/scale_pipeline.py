"""Large-scale tick resampling with Polars and Spark.

Binance trade dumps are large: a single liquid symbol is gigabytes per month,
and a multi-symbol study runs to hundreds of gigabytes. Resampling that with
pandas in memory is impractical, so this module provides two out-of-core
backends that produce the *same* OHLCV schema as
:func:`backtest.bars.resample_ohlcv`:

* :func:`resample_ohlcv_polars` — Polars lazy / streaming engine, good up to
  what one machine's disk and cores can stream.
* :func:`resample_ohlcv_spark` — PySpark, for distributed clusters.

Both imports are **lazy** (done inside the functions), so this module — and the
package — load without Polars or Spark installed. The pure-numpy/pandas core
and the test suite never import this file, which keeps the tested surface free
of heavy optional dependencies. Install them with the ``scale`` extra
(``pip install -e ".[scale]"``) or via the pixi environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl


def _polars_rule(rule: str) -> str:
    """Translate a pandas-style offset alias to a Polars duration string."""
    # Polars uses e.g. "1m", "5m", "1h", "1d"; pandas uses "1min", "1h", "1d".
    return rule.replace("min", "m")


def resample_ohlcv_polars(
    source: str | Any,
    rule: str,
    *,
    ts: str = "time",
    price: str = "price",
    size: str = "size",
    time_unit: str = "ms",
) -> pl.DataFrame:
    """Resample a large tick file to OHLCV bars using Polars (streaming).

    Parameters
    ----------
    source:
        A path/glob to CSV or Parquet tick files, or an existing Polars
        ``LazyFrame`` / ``DataFrame``.
    rule:
        Bar size as a pandas-style alias (``"1min"``, ``"5min"``, ``"1h"``);
        translated to Polars internally.
    ts, price, size:
        Column names for timestamp, price, and traded size.
    time_unit:
        Unit of an integer ``ts`` column (``"ms"`` for Binance epoch millis).

    Returns
    -------
    polars.DataFrame
        Columns ``ts, open, high, low, close, volume`` collected from the lazy
        plan with the streaming engine.
    """
    import polars as pl

    if isinstance(source, str):
        lf: pl.LazyFrame = (
            pl.scan_parquet(source)
            if source.endswith(".parquet")
            else pl.scan_csv(source)
        )
    elif isinstance(source, pl.LazyFrame):
        lf = source
    else:
        lf = source.lazy()

    lf = lf.with_columns(pl.col(ts).cast(pl.Datetime(time_unit=time_unit)).alias("_ts"))

    bars = (
        lf.sort("_ts")
        .group_by_dynamic("_ts", every=_polars_rule(rule), label="left", closed="left")
        .agg(
            pl.col(price).first().alias("open"),
            pl.col(price).max().alias("high"),
            pl.col(price).min().alias("low"),
            pl.col(price).last().alias("close"),
            pl.col(size).sum().alias("volume"),
        )
        .rename({"_ts": "ts"})
    )
    return bars.collect(streaming=True)


def resample_ohlcv_spark(
    df: Any,
    rule: str,
    *,
    ts: str = "time",
    price: str = "price",
    size: str = "size",
) -> Any:
    """Resample a tick Spark ``DataFrame`` to OHLCV bars (distributed).

    Uses a tumbling time window of ``rule`` over the ``ts`` column. ``open`` and
    ``close`` are taken as the first/last trade within each window ordered by
    time; ``high``/``low``/``volume`` are straightforward aggregates.

    Returns a Spark ``DataFrame`` with ``ts, open, high, low, close, volume``.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    window = F.window(F.col(ts), _spark_window(rule))
    order = Window.partitionBy(window).orderBy(F.col(ts))

    enriched = df.withColumn("_rn_first", F.row_number().over(order)).withColumn(
        "_rn_last", F.row_number().over(order.orderBy(F.col(ts).desc()))
    )
    bars = enriched.groupBy(window).agg(
        F.first(F.when(F.col("_rn_first") == 1, F.col(price)), ignorenulls=True).alias(
            "open"
        ),
        F.max(price).alias("high"),
        F.min(price).alias("low"),
        F.first(F.when(F.col("_rn_last") == 1, F.col(price)), ignorenulls=True).alias(
            "close"
        ),
        F.sum(size).alias("volume"),
    )
    return bars.select(
        F.col("window.start").alias("ts"),
        "open",
        "high",
        "low",
        "close",
        "volume",
    )


def _spark_window(rule: str) -> str:
    """Translate a pandas-style alias to a Spark window duration string."""
    # Spark wants e.g. "1 minute", "5 minutes", "1 hour".
    if rule.endswith("min"):
        n = rule[:-3] or "1"
        return f"{n} minutes"
    if rule.endswith("h"):
        n = rule[:-1] or "1"
        return f"{n} hours"
    if rule.endswith("d"):
        n = rule[:-1] or "1"
        return f"{n} days"
    return rule
