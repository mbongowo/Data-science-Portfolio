"""als-recommender: matrix-factorisation recommendation with honest evaluation.

This package fits an Alternating Least Squares matrix factorisation, scores it
against a popularity baseline with proper ranking metrics, and offers a Spark
MLlib ALS wrapper for data that does not fit in memory.

The package is split so that the interpretation-critical numeric core (the
pure-numpy ALS, the ranking metrics, the baseline, and the split) has no
third-party dependency beyond numpy/pandas and is always importable and
testable. The distributed path (:mod:`recsys.spark_als`) imports PySpark lazily
and is deliberately not pulled in by importing this package.
"""

from __future__ import annotations

from recsys.als import (
    als_factorize,
    als_factorize_biased,
    als_implicit,
    predict,
    predict_biased,
)
from recsys.baseline import popularity_scores, recommend_popular
from recsys.metrics import (
    average_precision_at_k,
    catalog_coverage,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from recsys.split import train_val_test_split

__all__ = [
    "als_factorize",
    "als_factorize_biased",
    "als_implicit",
    "predict",
    "predict_biased",
    "rmse",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision_at_k",
    "mean_reciprocal_rank",
    "catalog_coverage",
    "popularity_scores",
    "recommend_popular",
    "train_val_test_split",
    "__version__",
]

__version__ = "0.1.0"
