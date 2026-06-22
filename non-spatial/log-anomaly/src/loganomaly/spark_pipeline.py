"""Scale-out ingest/parse with Spark and an optional IsolationForest detector.

Loghub HDFS_v1 is ~11M lines; templating and counting it line by line in pure
Python is slow. This module does the same work at scale with PySpark and offers
an sklearn ``IsolationForest`` as an alternative ML detector. Both PySpark and
scikit-learn are **heavy** and pull a JVM / compiled wheels, so every import of
them happens *inside* a function. This module is never imported by the package
``__init__`` or the test suite, which keeps the core dependency-free.

The masking logic is identical to :mod:`loganomaly.templating`; only the engine
changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from loganomaly.templating import mask_line

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def spark_session(app_name: str = "log-anomaly") -> Any:
    """Create (or get) a local SparkSession. Imports pyspark lazily."""
    from pyspark.sql import SparkSession  # lazy: heavy, needs a JVM

    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()


def parse_logs_to_counts(
    glob: str,
    session_regex: str,
    out_parquet: str | Path,
    app_name: str = "log-anomaly",
) -> Path:
    """Parse a raw log glob into a per-session event-count Parquet with Spark.

    Each line is masked to a template (:func:`loganomaly.templating.mask_line`),
    the session key is pulled out with ``session_regex``, and the result is a
    session x template count table written to Parquet. Heavy imports are local.

    Parameters
    ----------
    glob:
        Spark-readable path / glob of raw log files.
    session_regex:
        Regex whose first capturing group is the session key (e.g. the block id).
    out_parquet:
        Destination Parquet path.
    app_name:
        Spark application name.

    Returns
    -------
    pathlib.Path
        The Parquet path written.
    """
    from pyspark.sql import functions as F  # noqa: N812 (sql convention)

    spark = spark_session(app_name)
    mask_udf = F.udf(mask_line)

    lines = spark.read.text(glob)
    sessioned = (
        lines.withColumn("session", F.regexp_extract("value", session_regex, 1))
        .where(F.col("session") != "")
        .withColumn("template", mask_udf(F.col("value")))
    )
    counts = sessioned.groupBy("session", "template").count()

    out = Path(out_parquet)
    counts.write.mode("overwrite").parquet(str(out))
    return out


def isolation_forest_scores(
    X: ArrayLike,
    *,
    contamination: float = 0.03,
    random_state: int = 42,
) -> NDArray[np.float64]:
    """Score sessions with an sklearn ``IsolationForest`` (lazy import).

    Returns the negated ``score_samples`` so that larger means more anomalous,
    matching the convention of the PCA reconstruction error.

    Parameters
    ----------
    X:
        ``(n_sessions, n_templates)`` event-count matrix.
    contamination:
        Expected anomaly fraction passed to the forest.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Length-``n_sessions`` anomaly scores (higher = more anomalous).
    """
    from sklearn.ensemble import IsolationForest  # lazy: heavy

    A = np.asarray(X, dtype=float)
    forest = IsolationForest(contamination=contamination, random_state=random_state)
    forest.fit(A)
    # score_samples: higher = more normal; negate so higher = more anomalous.
    return -np.asarray(forest.score_samples(A), dtype=float)
