"""Spark MLlib ALS wrapper for scale-out matrix factorisation.

The pure-numpy :mod:`recsys.als` is a faithful reference but holds the whole
ratings matrix in memory, which does not survive a real catalogue. This module
runs the same factorisation on Spark MLlib's distributed ALS, which scales to
the MovieLens-25M / Amazon-Reviews regime.

PySpark and a JVM are required only here. Every PySpark import is **lazy**
(inside the functions), so importing this module costs nothing and the rest of
the package — and the test suite — never needs Spark installed. Nothing in this
module runs at import time.

Typical use::

    from recsys.spark_als import build_session, train_als, recommend_for_users

    spark = build_session()
    model = train_als(ratings_sdf, rank=32, reg=0.1, iters=15, seed=0)
    recs = recommend_for_users(model, k=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.ml.recommendation import ALSModel
    from pyspark.sql import DataFrame, SparkSession


def build_session(
    app_name: str = "als-recommender", master: str = "local[*]"
) -> SparkSession:
    """Create (or get) a SparkSession.

    Parameters
    ----------
    app_name:
        Spark application name.
    master:
        Spark master URL. Defaults to ``local[*]`` (all local cores).

    Returns
    -------
    pyspark.sql.SparkSession
    """
    from pyspark.sql import SparkSession  # lazy import

    return SparkSession.builder.appName(app_name).master(master).getOrCreate()


def load_ratings(
    spark: SparkSession,
    path: str,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> DataFrame:
    """Load a ratings CSV into a Spark DataFrame with the expected dtypes.

    ALS needs integer user/item ids and a float rating. The selected columns are
    cast accordingly and renamed to ``user`` / ``item`` / ``rating``.
    """
    from pyspark.sql.functions import col  # lazy import

    sdf = spark.read.csv(path, header=True, inferSchema=True)
    return sdf.select(
        col(user_col).cast("int").alias("user"),
        col(item_col).cast("int").alias("item"),
        col(rating_col).cast("float").alias("rating"),
    )


def train_als(
    ratings: DataFrame,
    rank: int = 32,
    reg: float = 0.1,
    iters: int = 15,
    seed: int = 0,
) -> ALSModel:
    """Fit Spark MLlib ALS on an explicit-feedback ratings DataFrame.

    Parameters
    ----------
    ratings:
        Spark DataFrame with ``user`` (int), ``item`` (int), ``rating`` (float)
        columns, as produced by :func:`load_ratings`.
    rank, reg, iters, seed:
        ALS hyperparameters, matching :func:`recsys.als.als_factorize`:
        latent dimension, L2 regularisation, number of iterations, and the
        random seed.

    Returns
    -------
    pyspark.ml.recommendation.ALSModel
        ``coldStartStrategy="drop"`` so predictions for unseen users/items are
        dropped rather than returned as NaN.
    """
    from pyspark.ml.recommendation import ALS  # lazy import

    als = ALS(
        rank=rank,
        regParam=reg,
        maxIter=iters,
        seed=seed,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=False,
    )
    return als.fit(ratings)


def recommend_for_users(model: ALSModel, k: int = 10) -> DataFrame:
    """Return the top-``k`` item recommendations per user.

    Parameters
    ----------
    model:
        A fitted ``ALSModel``.
    k:
        Number of recommendations per user.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per user with a ``recommendations`` array of
        ``(item, rating)`` structs.
    """
    return model.recommendForAllUsers(k)


def predict(model: ALSModel, pairs: DataFrame) -> DataFrame:
    """Score given (user, item) pairs, returning a ``prediction`` column.

    Parameters
    ----------
    model:
        A fitted ``ALSModel``.
    pairs:
        Spark DataFrame with ``user`` and ``item`` columns.

    Returns
    -------
    pyspark.sql.DataFrame
        ``pairs`` with an added ``prediction`` column (cold-start rows dropped).
    """
    out: Any = model.transform(pairs)
    return out
