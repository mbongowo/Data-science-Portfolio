"""One-command, fully reproducible demo of the pure-numpy recommender core.

This drives the *real* numeric core end-to-end on a small seeded synthetic
ratings dataset, with no PySpark and nothing beyond numpy / pandas / pyyaml +
the standard library, so it runs anywhere (including CI) in a couple of seconds.

What it does, using the same functions the Spark pipeline and the CLI use:

* synthesise a small low-rank ratings matrix with noise
  (:func:`synthesize_ratings`), then hold out a per-user test set with the real
  :func:`recsys.split.train_val_test_split`;
* fit the real :func:`recsys.als.als_factorize` on the train split and compute
  the real popularity :func:`recsys.baseline.recommend_popular` baseline;
* score **both** on the held-out set with the real metrics
  (:func:`recsys.metrics.rmse`, :func:`~recsys.metrics.precision_at_k`,
  :func:`~recsys.metrics.recall_at_k`, :func:`~recsys.metrics.ndcg_at_k`);
* write artefacts (``metrics.csv``, ``topn_sample.csv``, ``summary.json``) and
  return a dict of both models' metrics plus the ALS lift over popularity.

Because the data is genuinely low rank, ALS measurably beats the
non-personalised popularity baseline on the ranking metrics — that lift is the
honest, reproducible headline number, not an illustrative placeholder.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from recsys.als import als_factorize, predict
from recsys.baseline import recommend_popular
from recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k, rmse
from recsys.split import train_val_test_split

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

# Demo hyperparameters. Chosen so the synthetic data is genuinely low rank and
# ALS measurably beats popularity, while the whole run stays well under a couple
# of seconds. Kept here (not magic numbers) so the demo is self-documenting.
N_USERS = 200
N_ITEMS = 100
TRUE_RANK = 3
DENSITY = 0.4  # fraction of the user-item matrix that is observed
NOISE = 0.15  # standard deviation of additive rating noise
RATING_LO, RATING_HI = 1.0, 5.0

ALS_RANK = 6
ALS_REG = 0.02
ALS_ITERS = 25
ALS_SEED = 0

VAL_RATIO = 0.0
TEST_RATIO = 0.2
SPLIT_SEED = 42

K = 10
RELEVANT_THRESHOLD = 4.0


def synthesize_ratings(seed: int = 0) -> pd.DataFrame:
    """Deterministically synthesise a small, sparse, low-rank ratings table.

    A true rank-``TRUE_RANK`` user/item factor pair generates a dense score
    matrix; that is rescaled to the rating range, perturbed with Gaussian noise,
    and then only a random ``DENSITY`` fraction of entries is kept as "observed"
    interactions. The result is a long DataFrame with ``user`` / ``item`` /
    ``rating`` columns, ready for the real split and matrix builders.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng``; fixing it makes the whole demo
        reproducible bit-for-bit.

    Returns
    -------
    pandas.DataFrame
        One row per observed interaction.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((N_USERS, TRUE_RANK))
    v = rng.standard_normal((N_ITEMS, TRUE_RANK))
    latent = u @ v.T  # (N_USERS, N_ITEMS), exactly rank TRUE_RANK

    # Rescale the latent scores to the rating range via their own min/max, then
    # add noise and clip back into range.
    lo, hi = latent.min(), latent.max()
    ratings = RATING_LO + (latent - lo) / (hi - lo) * (RATING_HI - RATING_LO)
    ratings = ratings + rng.normal(0.0, NOISE, size=ratings.shape)
    ratings = np.clip(ratings, RATING_LO, RATING_HI)

    observed = rng.random((N_USERS, N_ITEMS)) < DENSITY
    # Guarantee every user has at least a few interactions so the per-user split
    # and the warm-user evaluation are well defined.
    for i in range(N_USERS):
        if observed[i].sum() < 5:
            cols = rng.choice(N_ITEMS, size=5, replace=False)
            observed[i, cols] = True

    rows: list[tuple[int, int, float]] = []
    user_idx, item_idx = np.nonzero(observed)
    for ui, ii in zip(user_idx, item_idx, strict=False):
        rows.append((int(ui), int(ii), round(float(ratings[ui, ii]), 4)))
    return pd.DataFrame(rows, columns=["user", "item", "rating"])


def _build_matrix(
    train: pd.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[Any], list[Any]]:
    """Pivot a long ratings frame into a dense matrix ``R`` and observed mask."""
    users = sorted(train["user"].unique())
    items = sorted(train["item"].unique())
    u_index = {usr: i for i, usr in enumerate(users)}
    i_index = {it: j for j, it in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=float)
    mask = np.zeros_like(R, dtype=bool)
    for usr, it, r in zip(train["user"], train["item"], train["rating"], strict=False):
        R[u_index[usr], i_index[it]] = r
        mask[u_index[usr], i_index[it]] = True
    return R, mask, users, items


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the full synthetic demo end-to-end and write artefacts.

    Parameters
    ----------
    seed:
        Seed threaded through the data synthesis (the ALS and split seeds are
        fixed module constants so the headline numbers are stable).
    out_dir:
        Directory for the artefacts (created if missing): ``metrics.csv``,
        ``topn_sample.csv``, ``summary.json``.

    Returns
    -------
    dict
        ``k``, an ``als`` block and a ``popularity`` block (each with
        ``rmse``/``precision_at_k``/``recall_at_k``/``ndcg_at_k``), and a
        ``lift`` block of ALS-minus-popularity differences. ALS beats popularity
        on the ranking metrics on this low-rank synthetic data.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = synthesize_ratings(seed)
    train_df, _val, test_df = train_val_test_split(
        df,
        user_col="user",
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SPLIT_SEED,
    )

    R, mask, users, items = _build_matrix(train_df)
    u_index = {usr: i for i, usr in enumerate(users)}
    i_pos = {it: j for j, it in enumerate(items)}

    U, V = als_factorize(
        R, mask, rank=ALS_RANK, reg=ALS_REG, iters=ALS_ITERS, seed=ALS_SEED
    )
    scores = predict(U, V)

    # --- RMSE on the held-out ratings -------------------------------------
    # ALS predicts each held-out cell from its factors; popularity predicts the
    # item's mean training rating (its best non-personalised guess).
    item_mean = train_df.groupby("item")["rating"].mean().to_dict()
    global_mean = float(train_df["rating"].mean())

    als_true, als_pred = [], []
    pop_true, pop_pred = [], []
    for usr, it, r in zip(
        test_df["user"], test_df["item"], test_df["rating"], strict=False
    ):
        if usr in u_index and it in i_pos:
            als_true.append(r)
            als_pred.append(scores[u_index[usr], i_pos[it]])
        pop_true.append(r)
        pop_pred.append(item_mean.get(it, global_mean))

    als_rmse = rmse(als_true, als_pred)
    pop_rmse = rmse(pop_true, pop_pred)

    # --- Ranking metrics on warm users ------------------------------------
    relevant: dict[Any, set] = {}
    for usr, it, r in zip(
        test_df["user"], test_df["item"], test_df["rating"], strict=False
    ):
        if r >= RELEVANT_THRESHOLD:
            relevant.setdefault(usr, set()).add(it)

    # Items already seen by each user in training are excluded from both
    # recommenders, so the ranking task is to surface *new* relevant items.
    seen: dict[Any, set] = {}
    for usr, it in zip(train_df["user"], train_df["item"], strict=False):
        seen.setdefault(usr, set()).add(it)

    pop_p, pop_r, pop_n = [], [], []
    als_p, als_r, als_n = [], [], []
    topn_rows: list[dict[str, Any]] = []
    for usr in sorted(relevant):
        if usr not in u_index:
            continue
        rel = relevant[usr]
        grades = dict.fromkeys(rel, 1.0)
        already = seen.get(usr, set())

        pop_rec = recommend_popular(train_df, k=K, item_col="item", exclude=already)

        order = np.argsort(-scores[u_index[usr]])
        als_rec = [items[j] for j in order if items[j] not in already][:K]

        pop_p.append(precision_at_k(pop_rec, rel, K))
        pop_r.append(recall_at_k(pop_rec, rel, K))
        pop_n.append(ndcg_at_k(pop_rec, grades, K))
        als_p.append(precision_at_k(als_rec, rel, K))
        als_r.append(recall_at_k(als_rec, rel, K))
        als_n.append(ndcg_at_k(als_rec, grades, K))

        if len(topn_rows) < 5:
            topn_rows.append(
                {
                    "user": int(usr),
                    "als_topn": " ".join(str(int(i)) for i in als_rec),
                    "popularity_topn": " ".join(str(int(i)) for i in pop_rec),
                    "n_relevant": len(rel),
                }
            )

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    als_block = {
        "rmse": round(als_rmse, 4),
        "precision_at_k": round(_mean(als_p), 4),
        "recall_at_k": round(_mean(als_r), 4),
        "ndcg_at_k": round(_mean(als_n), 4),
    }
    pop_block = {
        "rmse": round(pop_rmse, 4),
        "precision_at_k": round(_mean(pop_p), 4),
        "recall_at_k": round(_mean(pop_r), 4),
        "ndcg_at_k": round(_mean(pop_n), 4),
    }
    lift = {
        name: round(als_block[name] - pop_block[name], 4)
        for name in ("rmse", "precision_at_k", "recall_at_k", "ndcg_at_k")
    }

    result: dict[str, Any] = {
        "k": K,
        "n_users": len(users),
        "n_items": len(items),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "als": als_block,
        "popularity": pop_block,
        "lift": lift,
    }

    # --- Artefacts --------------------------------------------------------
    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "popularity", "als", "lift"])
        for name in ("rmse", "precision_at_k", "recall_at_k", "ndcg_at_k"):
            w.writerow([name, pop_block[name], als_block[name], lift[name]])

    topn_path = out / "topn_sample.csv"
    with open(topn_path, "w", encoding="utf-8", newline="") as fh:
        fieldnames = ["user", "als_topn", "popularity_topn", "n_relevant"]
        w2 = csv.DictWriter(fh, fieldnames=fieldnames)
        w2.writeheader()
        w2.writerows(topn_rows)

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
