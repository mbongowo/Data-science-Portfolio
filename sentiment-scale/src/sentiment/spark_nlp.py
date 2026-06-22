"""Spark batch loading and batched model inference (lazy imports).

This is the "at scale" path: read the compressed Reddit dumps into a Spark
DataFrame, normalise the text, and score it either with the lexicon (the same
:func:`sentiment.lexicon.score_text`, applied as a UDF) or with a batched model
(VADER's compound score, evaluated per partition so the model is built once per
executor rather than once per row).

Everything here imports ``pyspark`` / ``vaderSentiment`` **inside** the
functions, so importing this module costs nothing and the test suite (which
never imports it) stays dependency-free. None of this is exercised by CI; it is
meant to run on a cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession


def get_spark(app_name: str = "sentiment-scale") -> "SparkSession":
    """Create (or get) a local SparkSession. Lazy import of pyspark."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def load_dumps(
    spark: "SparkSession",
    paths: list[str],
    *,
    text_field: str = "body",
) -> "DataFrame":
    """Load newline-JSON Reddit dumps into a DataFrame with ``text`` + ``date``.

    Parameters
    ----------
    spark:
        An active SparkSession.
    paths:
        Paths/globs to the (optionally compressed) JSON dumps. Spark reads
        ``.zst``/``.bz2`` transparently given the right Hadoop codecs.
    text_field:
        The JSON field carrying the document text (``body`` for comments).
    """
    from pyspark.sql import functions as F

    raw = spark.read.json(paths)
    return raw.select(
        F.col(text_field).alias("text"),
        F.from_unixtime(F.col("created_utc")).cast("timestamp").alias("date"),
    ).where(F.col("text").isNotNull())


def score_lexicon_spark(
    df: "DataFrame",
    lexicon: dict[str, float] | None = None,
    *,
    negation_window: int = 3,
    alpha: float = 15.0,
) -> "DataFrame":
    """Add a ``score`` column using the pure-python lexicon scorer as a UDF."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType

    from sentiment.lexicon import score_text

    def _score(text: str | None) -> float:
        if text is None:
            return 0.0
        return score_text(
            text, lexicon, negation_window=negation_window, alpha=alpha
        )

    score_udf = F.udf(_score, DoubleType())
    return df.withColumn("score", score_udf(F.col("text")))


def score_model_spark(df: "DataFrame", *, batch_size: int = 1024) -> "DataFrame":
    """Add a ``score`` column from VADER's compound score, batched per partition.

    The analyzer is constructed once per partition (inside
    ``mapPartitions``-style iteration) instead of once per row, which is what
    makes this usable on a multi-million-row corpus.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType

    def _score_partition(texts: "Any") -> "Any":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        for text in texts:
            yield analyzer.polarity_scores(text or "")["compound"]

    score_udf = F.pandas_udf(_score_partition, DoubleType())  # type: ignore[call-overload]
    return df.withColumn("score", score_udf(F.col("text")))


def write_parquet(df: "DataFrame", out_path: str) -> None:
    """Write the scored DataFrame to Parquet."""
    df.write.mode("overwrite").parquet(out_path)
