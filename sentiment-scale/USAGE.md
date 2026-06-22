# Usage guide: the sentiment-scale workflow

This guide walks through one pass of the pipeline: install the stack, load the
Reddit dumps to Parquet, clean the text, score it with the lexicon or a batched
model, validate the scorer against a labelled sample, aggregate to a sentiment
time series, and extract topics. It closes with what these scores do not
establish.

The pure-numpy/pandas core (`normalize_text`, `tokenize`, `score_text`,
`sentiment_timeseries`, `tfidf`) runs with only numpy and pandas installed and is
meant for small problems and for checking your understanding. The Spark batch
load and batched model inference that a multi-million-row corpus needs live in
`sentiment.spark_nlp`, which requires pyspark (and a JVM) as described below.

## 1. Install

The Spark/Arrow stack resolves most reliably through conda-forge; pyspark also
pulls a JVM. Pixi is the path the repository is set up for.

```bash
pixi install        # resolves dependencies and writes pixi.lock locally
pixi run test       # confirm the install: the test suite should pass
```

If you prefer pip:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A quick check that the numeric core imports without Spark or the model:

```bash
python -c "from sentiment import score_text, sentiment_timeseries, tfidf; print('ok')"
```

## 2. Get the data

The primary corpus is per-subreddit **Reddit Pushshift dumps** — compressed
newline JSON, one record per line. See [`data/README.md`](data/README.md) for
where to download them. Place one file per configured subreddit under
`data/raw/`, for example `data/raw/technology_comments.zst`.

Edit `config/reddit.yaml` to set the subreddits, the date range, and the text
field (`body` for comments). A large public Twitter/X set works as a drop-in
alternative; only the loader changes.

## 3. Ingest: dumps to cleaned Parquet

`ingest` reads the dumps with Spark, keeps the text field and the `created_utc`
timestamp, and writes Parquet. Parquet is columnar and compressed, so the score
and trends steps scan only the columns they need.

```bash
sentiment ingest --config config/reddit.yaml --out data/raw
```

This writes `data/raw/documents.parquet` with a `text` column and a `date`
timestamp. Decompression of `.zst`/`.bz2` is handled by Spark's input codecs.

## 4. Clean the text

Every scorer sees the same normalisation, so they agree on what a token is.
`normalize_text` lowercases, strips URLs and Markdown, drops punctuation
(replacing it with a space so words do not glue together), and collapses
whitespace; `tokenize` then splits on spaces.

```python
from sentiment.clean import normalize_text, tokenize

normalize_text("Check https://x.com -- it's GREAT!!!")   # 'check its great'
tokenize("Not great, really.")                           # ['not', 'great', 'really']
```

Documents with fewer than `clean.min_token_count` tokens are noisy for any
scorer; drop them before scoring (the Spark path filters on the token count, the
local path can filter on `len(tokenize(text))`).

## 5. Score: lexicon vs batched model

There are two scorers, chosen by `scoring.scorer` in the config.

**Lexicon (default).** `score_text` sums the valences of matched tokens, flips
the sign of any token inside the negation window after a negation word
(`not`/`no`/`never`), and squashes the sum `s` to roughly `[-1, 1]` with the
VADER-style `s / sqrt(s**2 + alpha)`:

```python
from sentiment.lexicon import score_text

score_text("great")       # +3 / sqrt(9 + 15)  ~= +0.61
score_text("not great")   # negation flips it  ~= -0.61
score_text("at noon")     # no lexicon hits    ==  0.0
```

It is fast, transparent, and language-explicit — and it misses context. Use it
as a coarse signal and a baseline.

**Model (`scorer: model`).** The model path scores VADER's compound score in
Spark, building the analyzer once per partition rather than once per row so it
scales:

```bash
sentiment score --config config/reddit.yaml --data data/raw --out outputs
```

This writes `outputs/scored.parquet` with a `score` column. The same command
runs either scorer depending on the config; the only difference is which branch
of `spark_nlp` it calls.

## 6. Validate on a labelled sample

A score series is worthless if the scorer is wrong, so validate before trusting
it. Hand-label (or borrow labels for) a few hundred documents as
positive/neutral/negative, run the scorer on them, threshold the continuous score
to a sign, and compare.

```python
import numpy as np

pred = np.sign([score_text(t) for t in sample_texts])   # -1 / 0 / +1
acc  = (pred == sample_labels).mean()                   # sign accuracy
```

Report sign accuracy and a macro-F1 (so a dominant class does not flatter the
number). Treat the lexicon's accuracy as a ceiling on how much weight any level
in the series can carry; if it only gets the sign right ~3/4 of the time, quote
*direction*, not magnitude. Validate the model path the same way and prefer it
where the two disagree.

## 7. Aggregate to a time series

`sentiment_timeseries` collapses the per-document scores to a mean per period.

```python
import pandas as pd
from sentiment.aggregate import sentiment_timeseries

df = pd.read_parquet("outputs/scored.parquet")
weekly = sentiment_timeseries(df, freq="weekly")   # columns: period, mean_score, n
```

`daily` uses calendar days; `weekly` uses pandas weeks (labelled by the
week-ending Sunday). Always carry the per-period document count `n`: a spike in a
thin week is noise, not signal. The CLI writes
`outputs/sentiment_timeseries.csv` and a `trends_summary.json`:

```bash
sentiment trends --config config/reddit.yaml --data outputs --out outputs
```

## 8. Topics: TF-IDF + clustering

To check whether a sentiment shift is really a *topic* shift, extract topics.
`tfidf` builds a `tf * idf` matrix with smoothed idf
`ln((N+1)/(df+1)) + 1` and a sorted vocabulary:

```python
from sentiment.topics import tfidf

matrix, vocab = tfidf(documents)   # (N x V) weights, deterministic column order
```

Cluster the rows (e.g. k-means from scikit-learn, `topics.n_clusters` clusters)
and read the top `topics.top_terms` terms per cluster by mean TF-IDF weight.
Overlay the cluster sizes on the weekly series: a sentiment dip that lines up
with a new topic cluster is a topic-mix effect, not necessarily a mood change.

## 9. How to interpret responsibly

These scores describe expressed sentiment in a few online communities. They do
not measure public opinion, and a few limits should travel with any result.

**Lexicon vs model.** A valence lexicon scores words, not meaning. It is fast and
auditable but blind to context the model can use. The two scorers disagree most
on ambiguous text; that disagreement is information — validate, and where they
split, trust the validated model.

**Sampling bias.** A handful of subreddits is not a population. Reddit skews
young, English-speaking, and self-selected by topic. The series is about *those
communities*, full stop.

**Sarcasm and irony.** "great, another outage" scores positive. Negation handling
catches "not great"; it does not catch sarcasm, and the error is not symmetric
across topics, so it can bias a comparison between communities.

**Topic drift.** A weekly mean can move because the *topic mix* moved, not because
sentiment did. The TF-IDF clusters exist to test that; read sentiment and topics
together, never the score alone.

**Not causal.** Sentiment dropping after an event is co-occurrence. Many things
share a week. Pattern is a prompt for explanation, not the explanation.

**Threshold and parameter sensitivity.** The sign of a near-zero score is fragile;
`min_token_count`, `negation_window`, and `alpha` all move borderline documents.
Report them, and check that the headline finding survives a reasonable change.
