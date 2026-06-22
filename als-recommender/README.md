# als-recommender

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Matrix-factorisation recommendation, evaluated honestly.** Fit an Alternating
Least Squares (ALS) model, then judge it the only way that means anything: does
it beat a non-personalised popularity baseline on proper ranking metrics, on
data the model never saw in training. A pure-numpy ALS reference makes the maths
checkable; a Spark MLlib ALS wrapper takes the same model to data that does not
fit in memory.

---

## Result first

**Question.** On **MovieLens-25M**, does a personalised ALS model actually beat
just recommending the globally most popular movies to everyone?

**Answer (illustrative).** Yes, but read the size of the lift before celebrating.
On a per-user 10/10 random holdout, with relevance defined as a held-out rating
of 4.0 or higher and a top-10 cut-off, ALS clears the popularity baseline on
every ranking metric — and predicts ratings (RMSE) well enough to be useful, but
RMSE is not what decides the ranking.

```
Metric (top-K = 10)        Popularity     ALS (rank=32)     Lift
RMSE  (rating error)            0.98            0.81        -0.17  (lower better)
Precision@10                    0.071           0.118       +0.047
Recall@10                       0.049           0.087       +0.038
NDCG@10                         0.083           0.142       +0.059
```

*(Numbers above are illustrative placeholders; run `make evaluate` to regenerate
them for the configured split, rank, and cut-off.)*

### What this result does **not** let you conclude

- **Offline lift is not online value.** Beating popularity on held-out clicks is
  not the same as helping a real user. Engagement, satisfaction, and revenue are
  measured with an A/B test, not a ranking metric.
- **Cold start is unsolved here.** ALS has no factors for a user or item it never
  saw in training. New users and new movies fall back to popularity; the headline
  metrics are computed on *warm* users and quietly ignore that gap.
- **Popularity bias.** Both the model and the evaluation lean on already-popular
  items. A held-out set drawn from logged behaviour rewards recommending what was
  already watched a lot, so the baseline is strong and the metrics flatter head
  items over the long tail.
- **Temporal leakage.** The default split is a *random* per-user holdout, so a
  training interaction can post-date a test one. That inflates the metrics versus
  a leakage-free time-ordered split. Use a temporal split for any claim that has
  to survive scrutiny.
- **One operating point.** Everything is reported at K = 10 and one relevance
  threshold. The ranking changes with both; a single row is not a curve.

---

## How it works

```
data/README.md         # how to fetch MovieLens-25M (ratings.csv) into data/raw
        |
src/recsys/
  split.py             # seeded per-user train/val/test holdout (disjoint, deterministic)
  als.py               # pure-numpy ALS: alternating ridge least squares + predict()
  baseline.py          # popularity ranking + recommend_popular (the yardstick)
  metrics.py           # rmse, precision@k, recall@k, ndcg@k (ranking quality)
  spark_als.py         # Spark MLlib ALS wrapper for scale (lazy pyspark import)
  cli.py               # `recsys` entry point: train / recommend / evaluate
```

The numeric core (`als.py`, `metrics.py`, `baseline.py`, `split.py`) is a pure
reference layer with no dependency beyond numpy/pandas. `als_factorize(R, mask,
rank, reg, iters, seed)` solves the explicit-feedback ALS objective by
alternating ridge regressions over the observed entries, and on a fully observed
low-rank matrix it reconstructs to a tiny RMSE. The metrics are covered by
**hand-derived known-answer tests**: RMSE is checked against `sqrt(5/3)`,
Precision@K and Recall@K against a worked one-hit example, and NDCG@K against a
case whose DCG/IDCG arithmetic is written out in the test docstring (a perfect
ranking gives exactly 1.0). The Spark MLlib path lives behind a lazy `pyspark`
import in `spark_als.py`, so the core and the test suite run without a JVM or
Spark installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: load ratings, split with
discipline, choose ALS hyperparameters, define the ranking metrics, compare
against the baseline, handle cold start, and a section on what the numbers do not
prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves PySpark + the JVM)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
# fetch MovieLens-25M into data/raw (see data/README.md), then:
pixi run train
pixi run evaluate
pixi run test
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make train
make evaluate
make test
```

### Option C — Docker

```bash
docker build -t als-recommender .
docker run --rm als-recommender        # runs the pure-numpy test suite
```

The Docker image runs the known-answer tests only; Spark MLlib training runs
separately (it needs a JVM and PySpark, supplied by the pixi/conda-forge env).

---

## Configuration

Everything analysis-defining lives in [`config/movielens.yaml`](config/movielens.yaml):
data path and column names, the train/val/test split ratios and seed, the ALS
hyperparameters (rank, reg, iters), and the evaluation cut-off `k` and relevance
threshold.

---

## Data sources

- **MovieLens-25M** (primary) — 25M ratings from ~162k users on ~62k movies, the
  standard collaborative-filtering benchmark. Free; see
  [`data/README.md`](data/README.md) for the fetch command.
- **Amazon Reviews 2023** (documented heavier option) — hundreds of millions of
  reviews, large and sparse enough to justify the Spark MLlib ALS path rather
  than the in-memory numpy reference. Only the loader changes; the pipeline is
  the same.

Raw data and outputs are git-ignored and regenerated from the fetch script and
the pipeline.

---

## License

MIT © 2026 Joseph Mbuh
