"""sentiment-scale: sentiment analysis of Reddit dumps at scale.

This package cleans text, scores it (a transparent lexicon scorer in the pure
core, or a batched model via Spark), validates the scorer on a labelled sample,
and aggregates scores into a daily/weekly sentiment time series, with TF-IDF
topic extraction on the side.

The package is split so that the interpretation-critical numeric core
(text cleaning, the lexicon scorer, time-series aggregation, and TF-IDF) has no
third-party dependency beyond numpy/pandas and is always importable and
testable. The heavy Spark + model-inference path lives in :mod:`sentiment.spark_nlp`
and is imported lazily, never by this module or the test suite.
"""

from __future__ import annotations

from sentiment.aggregate import sentiment_timeseries
from sentiment.clean import normalize_text, tokenize
from sentiment.lexicon import DEMO_LEXICON, score_text
from sentiment.topics import tfidf

__all__ = [
    "normalize_text",
    "tokenize",
    "score_text",
    "DEMO_LEXICON",
    "sentiment_timeseries",
    "tfidf",
    "__version__",
]

__version__ = "0.1.0"
