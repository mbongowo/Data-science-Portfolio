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

**Question.** Can the lexicon scorer recover a real sentiment trend shift, and
how trustworthy is its sign?

**Answer.** Yes. On a seeded synthetic corpus of short posts (each built from
known-valence words so its true label is known), the lexicon scorer reproduces
the sign of every labelled post — **validation accuracy 1.00** — and the
aggregation step recovers a planted inflection cleanly: the mean score flips from
**+0.26 before** the shift date to **−0.32 after** it. The weekly series sits near
**+0.20…+0.28** for the four weeks before the shift and drops to **negative
(down to −0.37)** in the weeks after.

```
Validation (labelled synthetic sample, n=336):  sign accuracy = 1.00
Mean lexicon score (synthetic posts, 2019)
  before 2019-03-29   :  +0.26
  on/after 2019-03-29 :  -0.32   (flip of 0.58)
Weekly mean score (selected weeks)
  2019-03-24 (pre)    :  +0.28
  2019-03-31 (cross)  :  +0.01
  2019-04-07 (post)   :  -0.37
Posts scored          :  336   (lexicon, negation_window=3, alpha=15)
```

**Defensible finding.** Weekly sentiment shifted from roughly **+0.28 to −0.37
around 2019-03-29** — a clear, reproducible flip that the lexicon scorer's sign
tracks with 100% accuracy on the labelled posts.

These are **real, reproducible numbers** from a small seeded **synthetic** corpus
(deterministic, runs in well under a second). The full Spark + NLP pipeline runs
the *same* scoring and aggregation on real Reddit Pushshift dumps — only the data
source and scale change.

### Reproduce

```bash
pixi run demo            # writes outputs/sentiment_timeseries.csv + summary.json
```

(Equivalently `make demo`, or `sentiment demo --out outputs`. Needs only
numpy/pandas/pyyaml + stdlib — no Spark, no model.)

### What this analysis does **not** let you conclude

- **Lexicon vs model.** A valence lexicon scores words, not meaning. It misses
  context a *learned* model can catch, and the two disagree most on ambiguous
  text. The pure-numpy `classify.LogisticRegression` is a trained alternative you
  can fit and validate against in-repo (the configured Spark `scorer: model` path
  scales the same idea); treat the lexicon series as a coarse signal and validate
  before quoting a level.
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
  classify.py        # LogisticRegression + bag_of_words: a TRAINED model scorer
  aggregate.py       # sentiment_timeseries: mean score per day/week (pandas)
  uncertainty.py     # bootstrap_mean_ci: confidence interval for a mean
  topics.py          # tfidf + nmf: TF-IDF and NMF topic modelling (numpy)
  spark_nlp.py       # Spark batch load + batched model inference (lazy imports)
  cli.py             # `sentiment` console entry point: ingest / score / trends
```

The numeric core (`clean`, `lexicon`, `classify`, `aggregate`, `uncertainty`,
`topics`) is pure numpy/pandas/stdlib with no Spark or model dependency, so it is
always importable and always tested. It is covered by **hand-derived
known-answer and property-based tests** whose expected values are computed by
hand or pinned to a documented property:

- **Lexicon scorer** returns *+3/√24* for "great" and *−3/√24* for "not great"
  (the negation window flips the sign).
- **Trained classifier** (`classify.LogisticRegression`) is a pure-numpy logistic
  regression fit by gradient descent on bag-of-words features — the *learned*
  alternative to the fixed lexicon. It drives its cross-entropy loss down
  monotonically and reaches **100% train accuracy on a linearly separable** toy
  set; `predict_proba` / `predict` give probabilities and hard labels.
- **Bootstrap CIs** (`uncertainty.bootstrap_mean_ci`) attach a seeded
  percentile-bootstrap confidence interval to a mean sentiment; the test pins
  that the 95% interval **brackets the true mean** on a fixed sample.
- **Aggregation** returns means worked out on a three-row frame.
- **TF-IDF** returns *tf · (ln(3/2)+1)* on a two-document toy corpus, and
  **NMF** (`topics.nmf`, Lee–Seung multiplicative updates) returns **non-negative
  factors** whose reconstruction error is **monotone non-increasing**.

The heavy Spark + model-inference path lives in `spark_nlp.py` with lazy imports
and is never touched by the test suite, so the core and CI run without pyspark,
vaderSentiment, or a JVM installed.

A short, runnable tour of all of this lives in
[`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb): it runs the
demo, compares the lexicon against the trained model, shows a bootstrap CI, and
prints NMF topics. Rebuild it with `PYTHONPATH=src python notebooks/build_walkthrough.py`.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: dumps to Parquet, text
cleaning, lexicon vs batched-model scoring, validation on a labelled sample,
time-series aggregation, topic extraction, and a section on what these scores do
not prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves Spark/Arrow)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run demo           # seeded synthetic end-to-end demo (no dumps needed)
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
make demo
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
