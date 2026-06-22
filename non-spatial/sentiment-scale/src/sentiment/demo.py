"""One-command, dependency-light demo of the real sentiment-scale core.

This drives the *actual* pure-numpy/pandas core
(:func:`sentiment.clean.tokenize`, :func:`sentiment.lexicon.score_text`,
:func:`sentiment.aggregate.sentiment_timeseries`, :func:`sentiment.topics.tfidf`)
end-to-end on a small, deterministically synthesised, labelled set of short
posts built from the built-in :data:`sentiment.lexicon.DEMO_LEXICON` vocabulary.

Because every post is assembled from known-valence words, its true sentiment
label (positive / negative / neutral) is known, so the lexicon scorer can be
*validated* against ground truth (sign agreement). A deliberate sentiment trend
shift is planted around a fixed date so the aggregation step has a real
inflection to surface.

It depends only on numpy / pandas / stdlib, so it runs anywhere — including CI —
without pyspark or vaderSentiment. The full Spark + NLP pipeline
(:mod:`sentiment.spark_nlp`) runs this *same* scoring and aggregation on real
Reddit Pushshift dumps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment.aggregate import sentiment_timeseries
from sentiment.lexicon import DEMO_LEXICON, score_text
from sentiment.topics import tfidf

#: Filler tokens with no lexicon valence, used to pad posts so they read like
#: short sentences without changing their (sign-of-) sentiment.
_FILLER: tuple[str, ...] = (
    "the",
    "this",
    "today",
    "really",
    "was",
    "is",
    "i",
    "think",
    "it",
    "and",
)

#: Positive- and negative-valence words drawn from DEMO_LEXICON, plus the
#: negated constructions the scorer's negation window must handle.
_POS_PHRASES: tuple[str, ...] = ("good", "great", "not bad", "not terrible")
_NEG_PHRASES: tuple[str, ...] = ("bad", "terrible", "not good", "not great")


def _true_label(score: float) -> int:
    """Sign of a score as -1 / 0 / +1 (ground-truth or predicted class)."""
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Synthesise labelled posts, run the real core, and write artifacts.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng``; the whole run is deterministic.
    out_dir:
        Directory for the written artifacts
        (``sentiment_timeseries.csv`` and ``summary.json``). Created if absent.

    Returns
    -------
    dict
        ``num_posts``, ``validation_accuracy``, ``mean_sentiment_before``,
        ``mean_sentiment_after``, ``shift_date`` and ``top_terms``.

    Notes
    -----
    The posts span a fixed date range. Before ``shift_date`` the corpus is
    positive-leaning; on and after it the corpus flips negative-leaning. The
    returned before/after means recover that planted inflection from the
    aggregated weekly series, driven entirely by the real lexicon scorer.
    """
    rng = np.random.default_rng(seed)

    # --- Fixed experiment design (deterministic, independent of `seed`) ----
    start = pd.Timestamp("2019-03-01")
    n_days = 56  # eight weeks
    shift_date = pd.Timestamp("2019-03-29")  # planted inflection (week 5)
    posts_per_day = 6

    # Pre-shift days lean positive (80% positive phrases); post-shift days lean
    # negative (80% negative phrases). The flip is the signal to recover.
    rows: list[dict[str, Any]] = []
    for day_offset in range(n_days):
        date = start + pd.Timedelta(days=day_offset)
        post_shift = date >= shift_date
        p_pos = 0.2 if post_shift else 0.8
        for _ in range(posts_per_day):
            positive = rng.random() < p_pos
            phrase = (
                _POS_PHRASES[rng.integers(len(_POS_PHRASES))]
                if positive
                else _NEG_PHRASES[rng.integers(len(_NEG_PHRASES))]
            )
            n_fill = int(rng.integers(2, 5))
            fillers = [_FILLER[rng.integers(len(_FILLER))] for _ in range(n_fill)]
            # Interleave the sentiment phrase among filler words.
            cut = int(rng.integers(0, n_fill + 1))
            words = fillers[:cut] + phrase.split() + fillers[cut:]
            text = " ".join(words)
            # Ground-truth label is known by construction (which phrase pool the
            # post was drawn from), independent of the scorer under test.
            rows.append({"date": date, "text": text, "label": 1 if positive else -1})

    posts = pd.DataFrame(rows)

    # --- Drive the REAL core: clean + lexicon score every post -------------
    posts["score"] = posts["text"].map(lambda t: score_text(t, DEMO_LEXICON))

    # --- Validate the scorer against known labels (sign agreement) ---------
    # Ground-truth label was recorded at synthesis time (the phrase pool the
    # post was drawn from), so validation is independent of the scorer.
    pred = posts["score"].map(_true_label)
    validation_accuracy = float((posts["label"] == pred).mean())

    # --- Aggregate into a weekly sentiment time series ---------------------
    weekly = sentiment_timeseries(posts[["date", "score"]], freq="weekly")

    # --- Surface the planted inflection: mean before vs on/after shift -----
    before_mask = posts["date"] < shift_date
    mean_before = float(posts.loc[before_mask, "score"].mean())
    mean_after = float(posts.loc[~before_mask, "score"].mean())

    # --- TF-IDF top terms over the corpus (real topics.tfidf) --------------
    matrix, vocab = tfidf(posts["text"].tolist())
    term_weights = matrix.sum(axis=0)
    order = np.argsort(term_weights)[::-1]
    top_terms = [vocab[i] for i in order[:8]]

    # --- Write artifacts ---------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path / "sentiment_timeseries.csv", index=False)

    result: dict[str, Any] = {
        "num_posts": int(len(posts)),
        "validation_accuracy": round(validation_accuracy, 6),
        "mean_sentiment_before": round(mean_before, 6),
        "mean_sentiment_after": round(mean_after, 6),
        "shift_date": shift_date.strftime("%Y-%m-%d"),
        "top_terms": top_terms,
    }
    with open(out_path / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result
