# Usage guide: the recommendation workflow

This guide walks through one pass with this repository: install the stack, load
the ratings, split them with discipline, fit ALS, define the ranking metrics,
compare against a popularity baseline, deal with cold start, and read the result
honestly. It closes with what these numbers do not establish.

The pure-numpy reference functions (`als_factorize`, `predict`,
`als_factorize_biased`, `predict_biased`, `als_implicit`, `rmse`,
`precision_at_k`, `recall_at_k`, `ndcg_at_k`, `average_precision_at_k`,
`mean_reciprocal_rank`, `catalog_coverage`, `train_val_test_split`,
`popularity_scores`, `recommend_popular`) run with only numpy and pandas
installed and are meant for small problems and for checking your understanding.
The distributed training that real catalogues need lives in `recsys.spark_als`,
which requires PySpark and a JVM. A runnable tour of the explicit / biased /
implicit models and the extra ranking metrics is in
[`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb).

## 1. Install

PySpark wants a JVM on the PATH; conda-forge supplies both cleanly, so pixi is
the path the repository is set up for.

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

A quick check that the numeric core is importable without Spark:

```bash
python -c "import numpy, pandas; from recsys import als_factorize, ndcg_at_k; print('ok')"
```

## 2. Get the data

The primary dataset is MovieLens-25M. Fetch it into `data/raw/` as described in
[`data/README.md`](data/README.md); you want `data/raw/ml-25m/ratings.csv`, which
is the path `config/movielens.yaml` points at. For a quick local run, the
100k-rating `ml-latest-small` archive is a drop-in substitute.

## 3. Load the ratings

The pipeline works on a long table with one row per interaction and three
canonical columns: `user`, `item`, `rating`. The CLI renames the MovieLens
columns for you according to the config:

```python
import pandas as pd

df = pd.read_csv("data/raw/ml-25m/ratings.csv").rename(
    columns={"userId": "user", "movieId": "item", "rating": "rating"}
)
```

Explicit feedback (a 0.5–5.0 star rating) is what `als_factorize` is built for.
If you only have implicit feedback (clicks, plays), the right objective is
weighted ALS with a confidence term. The pure-numpy reference for that is
`als_implicit` (the Hu-Koren-Volinsky formulation, covered below); at scale it
is the Spark MLlib `implicitPrefs=True` path.

## 4. Split with discipline

A recommender must be judged on interactions it never trained on, and the split
has to respect users so each user can appear in train and in the held-out sets.

```python
from recsys.split import train_val_test_split

train, val, test = train_val_test_split(
    df, user_col="user", val_ratio=0.1, test_ratio=0.1, seed=42
)
```

The split is a **seeded per-user random holdout**: each user's rows are shuffled
deterministically and the tail fractions go to validation and test. It is
disjoint (no row appears twice), deterministic (a fixed seed reproduces it), and
per-user (a user's held-out rows come only from their own history). Two cautions:

- **Flooring keeps small users in train.** A user with very few ratings holds out
  nothing, so they are warm at evaluation but contribute no held-out rows.
- **Random holdout can leak across time.** A training interaction may post-date a
  test one. For a leakage-free evaluation, replace this with a temporal split on
  the timestamp column and re-run; expect the metrics to drop, because they were
  partly reading the future.

Tune hyperparameters on `val`. Touch `test` once, at the end.

## 5. Fit ALS

ALS factorises the (sparse, mostly missing) ratings matrix `R` into user factors
`U` and item factors `V` so that `U @ V.T` approximates the observed ratings.
Only the entries flagged by the boolean `mask` are fit.

```python
import numpy as np
from recsys.als import als_factorize, predict

# build a dense R + observed mask from the training rows (CLI does this for you)
U, V = als_factorize(R, mask, rank=32, reg=0.1, iters=15, seed=0)
scores = predict(U, V)        # dense (n_users x n_items) score matrix
```

The three hyperparameters that matter:

| Hyperparameter | What it controls | Symptom if wrong |
|---|---|---|
| `rank` | Latent dimension — how much structure the factors can capture | Too low underfits (everything looks average); too high overfits and is slow |
| `reg` (`lambda`) | L2 shrinkage on the factors | Too low overfits sparse users/items; too high washes out personalisation toward the mean |
| `iters` | Number of alternating sweeps | Too few stops short of convergence; past convergence it is wasted compute |

Each sweep solves, in closed form, a ridge regression per user (holding items
fixed) and then per item (holding users fixed). Alternating these never increases
the objective, so it converges to a local optimum. On a fully observed low-rank
matrix the reference reconstructs `R` to a tiny RMSE — that is the known-answer
test.

### Biased matrix factorisation

Real ratings are often dominated by *additive* effects — a user who rates
everything high, an item everyone loves — rather than interaction structure. The
biased model factors those offsets out explicitly,

```
R̂[i,j] = μ + bᵤ[i] + bᵥ[j] + U[i]·V[j]
```

so the latent factors no longer have to spend dimensions approximating the
offsets:

```python
from recsys.als import als_factorize_biased, predict_biased

mu, bu, bv, U, V = als_factorize_biased(R, mask, rank=32, reg=0.1, iters=15, seed=0)
scores = predict_biased(mu, bu, bv, U, V)
```

The global mean `μ` is the mean of the observed ratings; each sweep then updates
the user biases, item biases, user factors, and item factors with closed-form
ridge solves. On data whose signal is mostly additive offsets, the biased model
reaches a lower RMSE than the unbiased one at the same rank — that improvement is
the known-answer test.

### Implicit feedback (Hu-Koren-Volinsky)

With implicit feedback you only observe *that* an interaction happened, not how
much it was liked. The HKV objective turns counts into a binary preference `P`
(1 if any interaction) and a per-cell confidence `C` (how strongly to trust it),
then fits **every** cell weighted by confidence:

```python
import numpy as np
from recsys.als import als_implicit, predict

alpha = 40.0
C = 1.0 + alpha * counts        # confidence grows with interaction count
P = (counts > 0).astype(float)  # binary preference
U, V = als_implicit(C, P, rank=32, reg=0.1, iters=15, seed=0)
scores = predict(U, V)          # rank each user's row for a top-N list
```

The zeros are real low-confidence "no preference" signal, not missing data, so
unlike the explicit model there is no mask. On a clean two-block preference
matrix (two user groups, disjoint preferred items) the learned scores separate
each group's block from the other — the known-answer test.

## 6. Ranking metrics

Recommendation is a ranking problem, so the metrics score an *ordered* top-K
list, not a single rating.

```python
from recsys.metrics import rmse, precision_at_k, recall_at_k, ndcg_at_k

# recommended: ordered list of item ids, best first
# relevant: set of items the user actually liked in the held-out split
precision_at_k(recommended, relevant, k=10)   # fraction of top-10 that are relevant
recall_at_k(recommended, relevant, k=10)      # fraction of relevant items caught in top-10
ndcg_at_k(recommended, relevance, k=10)       # order-sensitive, normalised to [0, 1]
```

Definitions, precisely:

- **RMSE** — `sqrt(mean((y_true - y_pred)^2))`. The regression view: how close the
  predicted ratings are. It does **not** measure ranking; a model can have a good
  RMSE and still order the top-K badly.
- **Precision@K** — `hits / k`, where `hits` is the number of the top-K
  recommendations that are relevant. The denominator is `k` even for short lists.
- **Recall@K** — `hits / |relevant|`. Of everything the user liked, how much did
  the top-K catch. Zero by definition when the user has no relevant items.
- **NDCG@K** — discounted cumulative gain, normalised. Gain at rank `r` (starting
  at 1) is `relevance[item] / log2(r + 1)`, so rank 1 has discount `1/log2(2) = 1`.
  `NDCG = DCG / IDCG`, where IDCG is the DCG of the ideal (relevance-sorted)
  ordering. A perfect ranking is exactly 1.0. NDCG is the one that rewards putting
  the best items *first*, not merely somewhere in the list.

Relevance is graded from the held-out ratings: a held-out rating at or above the
configured threshold (default 4.0) is relevant.

Three more metrics round out the picture (each with a hand-derived known-answer
test):

```python
from recsys.metrics import (
    average_precision_at_k, mean_reciprocal_rank, catalog_coverage,
)

average_precision_at_k(recommended, relevant, k=10)  # per-user MAP building block
mean_reciprocal_rank(rec_lists, relevant_sets)       # 1/rank of the first hit, averaged
catalog_coverage(rec_lists, catalog)                 # fraction of items ever recommended
```

- **Average Precision@K** — averages the running precision at each relevant hit
  in the top-K, normalised by `min(|relevant|, k)`. The mean of this over users
  is **MAP**, the headline order-sensitive accuracy number.
- **MRR** — for each user, `1 / rank` of the *first* relevant item (0 if none),
  averaged over users. The right lens when the user only looks at the top result.
- **Catalogue coverage** — the fraction of the catalogue that is recommended to
  *someone*. A diversity / aggregate-fairness check: accuracy can be high while
  the system keeps surfacing the same few head items, and coverage exposes that.

## 7. Compare against the popularity baseline

The number that matters is not ALS in absolute terms but ALS *minus* a baseline
that took no effort. Popularity recommends the globally most-watched items to
everyone:

```python
from recsys.baseline import recommend_popular

pop_list = recommend_popular(train, k=10, item_col="item")   # same list for all users
```

Score both the ALS lists and `pop_list` with the metrics above, averaged over
held-out users. Run the whole comparison from the CLI:

```bash
recsys evaluate --config config/movielens.yaml
```

This writes `outputs/evaluation.json` with Precision@K, Recall@K, and NDCG@K for
both ALS and popularity. Popularity is a deceptively strong baseline on
behaviour-logged data; if ALS does not clearly beat it, the rank/reg are
mistuned, the split is leaking, or personalisation genuinely is not paying off
on this slice.

## 8. Cold start

ALS only has factors for users and items seen in training. At recommendation
time:

- **New user, no history** — there is no `U` row to score. Fall back to the
  popularity list until the user has enough interactions to factorise.
- **New item, never rated** — there is no `V` row, so it can never be recommended
  by ALS. Bootstrap it with content features or surface it through a popularity /
  exploration slot.
- **Warm-but-thin users** — a user with a handful of ratings gets a noisy factor;
  `reg` is what keeps that factor from chasing the noise.

The headline metrics are computed on warm users, so they silently exclude the
cold-start population. State that population's size alongside the metrics, or the
comparison flatters the model.

## 9. Scale out with Spark MLlib

When the ratings matrix no longer fits in memory, the same model runs on Spark
MLlib's distributed ALS. PySpark is imported lazily, so this path is opt-in:

```python
from recsys.spark_als import build_session, load_ratings, train_als, recommend_for_users

spark = build_session()
ratings = load_ratings(spark, "data/raw/ml-25m/ratings.csv")
model = train_als(ratings, rank=32, reg=0.1, iters=15, seed=0)
recs = recommend_for_users(model, k=10)
```

The objective and hyperparameters match the numpy reference; only the execution
engine changes. `coldStartStrategy="drop"` means predictions for unseen
users/items are dropped rather than returned as NaN, which keeps the evaluation
honest about what the model can actually score.

## 10. How to interpret responsibly

These metrics describe agreement with logged behaviour. They do not measure the
value of a recommendation, and a few limits should travel with any result.

**Offline is not online.** Beating popularity on held-out interactions is not the
same as helping a user. The decision-grade evidence is an A/B test on engagement
or revenue; offline metrics are a filter for which models are worth testing.

**The baseline is strong on purpose.** Popularity exploits the same behaviour bias
the held-out set is drawn from. A small lift over it is a real result, not a
disappointment; a model that cannot beat it has not earned its complexity.

**Leakage inflates everything.** A random holdout lets training peek at the
future relative to test. Re-run with a temporal split before quoting a number you
have to defend, and report which split produced it.

**One operating point is not a curve.** K and the relevance threshold both move
the ranking metrics. Report the cut-off you used and, ideally, a small sweep
rather than a single cell.

**Popularity bias compounds.** Optimising and evaluating on logged behaviour
rewards head items and starves the long tail. If catalogue coverage or fairness
matters, measure them directly; accuracy metrics will not show the harm.
