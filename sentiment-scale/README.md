# sentiment-scale

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Sentiment analysis at scale on Reddit dumps**: load multi-year Pushshift
archives into Parquet, clean the text, score it with a transparent lexicon (or a
batched model on Spark), validate the scorer on a labelled sample, and aggregate
to a daily/weekly sentiment series with TF-IDF topic extraction. What the scores
do and do not support is written out, not hand-waved.

---

## Result first

**Question.** Did weekly sentiment in **r/technology (2019)** shift around a
specific event — say a major platform's data-breach disclosure?

**Answer (illustrative).** Yes, and modestly. The weekly mean lexicon score sits
near a mild-positive baseline for most of the year, then drops for the two weeks
following the disclosure before recovering. The lexicon scorer reproduces the
sign of a hand-labelled sample on **~78%** of documents, so the *direction* of
the shift is trustworthy even if the absolute level is not.

![Placeholder weekly sentiment series](outputs/.gitkeep)
<!-- Running `make trends` writes outputs/sentiment_timeseries.csv and
     trends_summary.json; plot the weekly series and drop the PNG here. -->

```
Validation (labelled sample, n=400):  sign accuracy = 0.78   macro-F1 = 0.71
Weekly mean score (r/technology, 2019)
  baseline (wk 01-30)      :  +0.12
  event window (wk 31-32)  :  -0.05   (drop of 0.17)
  recovery (wk 33-40)      :  +0.09
Documents scored           :  1.42M   (lexicon, negation_window=3, alpha=15)
```

*(Numbers above are illustrative placeholders; run the pipeline to regenerate
them for the configured subreddits and window.)*

### What this analysis does **not** let you conclude

- **Lexicon vs model.** A valence lexicon scores words, not meaning. It misses
  context the configured model path (`scorer: model`) can catch, and the two
  disagree most on ambiguous text. Treat the lexicon series as a coarse signal;
  validate against the model before quoting a level.
- **Sampling bias.** A few subreddits are not "the public". Reddit skews young,
  English-speaking, and topic-selected. The series describes *those communities*,
  not a population.
- **Sarcasm and irony.** "great, another outage" reads positive to the lexicon
  and to most lightweight models. Negation handling catches "not great", not
  sarcasm, so error is not symmetric across topics.
- **Topic drift.** A change in the weekly score can be a change in *what is being
  discussed*, not in *how people feel*. The TF-IDF topic clusters are there to
  check whether a sentiment shift coincides with a topic shift; read them
  together.
- **Not causal.** A dip after an event is co-occurrence, not proof the event
  caused it. Other things happen in the same week.

---

## How it works

```
sentiment ingest     # compressed Reddit dumps -> cleaned Parquet (Spark)
        |
src/sentiment/
  clean.py           # normalize_text / tokenize: lowercase, strip URLs/markdown
  lexicon.py         # score_text: VADER-style valence sum + negation + squash
  aggregate.py       # sentiment_timeseries: mean score per day/week (pandas)
  topics.py          # tfidf: tf * smoothed-idf for topic extraction (numpy)
  spark_nlp.py       # Spark batch load + batched model inference (lazy imports)
  cli.py             # `sentiment` console entry point: ingest / score / trends
```

The numeric core (`clean`, `lexicon`, `aggregate`, `topics`) is pure
numpy/pandas/stdlib with no Spark or model dependency, so it is always importable
and always tested. It is covered by **hand-derived known-answer tests** whose
expected values are computed by hand: the lexicon scorer returns
*+3/√24* for "great" and *−3/√24* for "not great" (the negation window flips the
sign); daily/weekly aggregation returns means worked out on a three-row frame;
and TF-IDF returns *tf · (ln(3/2)+1)* on a two-document toy corpus. The heavy
Spark + model-inference path lives in `spark_nlp.py` with lazy imports and is
never touched by the test suite, so the core and CI run without pyspark,
vaderSentiment, or a JVM installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: dumps to Parquet, text
cleaning, lexicon vs batched-model scoring, validation on a labelled sample,
time-series aggregation, topic extraction, and a section on what these scores do
not prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves Spark/Arrow)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run ingest         # needs Reddit dumps under data/raw (see data/README.md)
pixi run score
pixi run trends
pixi run test
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make ingest
make score
make trends
make test
```

### Option C — Docker

```bash
docker build -t sentiment-scale .
docker run --rm sentiment-scale        # runs the test suite
```

---

## Configuration

Everything analysis-defining lives in [`config/reddit.yaml`](config/reddit.yaml):
the subreddits and date range, the text field, the cleaning threshold
(`min_token_count`), the scorer (`lexicon` | `model`) and its parameters, the
aggregation frequency (`daily` | `weekly`), and the topic-extraction settings.

---

## Data sources

- **Reddit Pushshift dumps** (primary) — per-subreddit comment/submission
  archives as compressed newline JSON. Free, public, multi-year, which is what
  makes a sentiment *time series* possible. See
  [`data/README.md`](data/README.md) for how to obtain a subreddit dump.
- **Large public Twitter/X academic set** (documented alternative) — e.g. the
  Sentiment140 1.6M-tweet corpus or an academic-access X archive. The pipeline
  is identical; only the loader changes, since a document is just
  `(text, timestamp)`.

Raw dumps and outputs are git-ignored and regenerated by the pipeline.

---

## License

MIT © 2026 Joseph Mbuh
