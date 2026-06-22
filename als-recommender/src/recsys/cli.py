"""Command-line entry point for als-recommender.

Subcommands drive the workflow from a YAML config:

    recsys train     --config config/movielens.yaml   # fit ALS factors
    recsys recommend --config config/movielens.yaml   # write top-N lists
    recsys evaluate  --config config/movielens.yaml   # ALS vs popularity
    recsys demo                                        # synthetic end-to-end demo

The ``demo`` subcommand needs no config or data: it synthesises a small seeded
low-rank ratings set and drives the real numpy core (ALS, baseline, metrics)
end-to-end in seconds, with no PySpark.

The heavy imports (pandas, the Spark wrapper) happen inside the command bodies
so that importing this module never requires the full stack. ``typer`` itself is
imported lazily inside :func:`main` for the same reason: the pure-numpy core and
the test suite never import this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_ratings(cfg: dict[str, Any]) -> Any:
    """Load the ratings CSV into a pandas DataFrame with canonical columns."""
    import pandas as pd

    dcfg = cfg["data"]
    df = pd.read_csv(dcfg["ratings_path"])
    return df.rename(
        columns={
            dcfg["user_col"]: "user",
            dcfg["item_col"]: "item",
            dcfg["rating_col"]: "rating",
        }
    )


def _build_matrix(train: Any) -> tuple[Any, Any, list[Any], list[Any]]:
    """Pivot a long ratings frame into a dense matrix R and observed mask."""
    import numpy as np

    users = sorted(train["user"].unique())
    items = sorted(train["item"].unique())
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {it: j for j, it in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=float)
    mask = np.zeros_like(R, dtype=bool)
    for u, it, r in zip(train["user"], train["item"], train["rating"], strict=False):
        R[u_index[u], i_index[it]] = r
        mask[u_index[u], i_index[it]] = True
    return R, mask, users, items


def train(config_path: str | Path) -> dict[str, Any]:
    """Fit ALS factors on the training split and save them to the output dir."""
    import numpy as np

    from recsys.als import als_factorize
    from recsys.split import train_val_test_split

    cfg = _load_config(config_path)
    df = _load_ratings(cfg)
    scfg, acfg = cfg["split"], cfg["als"]

    train_df, _val, _test = train_val_test_split(
        df,
        user_col="user",
        val_ratio=scfg["val_ratio"],
        test_ratio=scfg["test_ratio"],
        seed=scfg["seed"],
    )
    R, mask, users, items = _build_matrix(train_df)
    U, V = als_factorize(
        R,
        mask,
        rank=acfg["rank"],
        reg=acfg["reg"],
        iters=acfg["iters"],
        seed=acfg["seed"],
    )

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "als_factors.npz",
        U=U,
        V=V,
        users=np.array(users),
        items=np.array(items),
    )
    summary = {"n_users": len(users), "n_items": len(items), "rank": acfg["rank"]}
    print(json.dumps(summary, indent=2))
    return summary


def recommend(config_path: str | Path) -> dict[str, Any]:
    """Write top-N recommendations per user from saved ALS factors."""
    import numpy as np

    from recsys.als import predict

    cfg = _load_config(config_path)
    k = cfg["eval"]["k"]
    out = Path(cfg["output"]["dir"])
    data = np.load(out / "als_factors.npz", allow_pickle=True)
    U, V, items = data["U"], data["V"], data["items"]

    scores = predict(U, V)
    top = np.argsort(-scores, axis=1)[:, :k]
    recs = {int(u): [items[j].item() for j in row] for u, row in enumerate(top)}
    with open(out / "recommendations.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=2)
    summary = {"n_users": len(recs), "k": k}
    print(json.dumps(summary, indent=2))
    return summary


def evaluate(config_path: str | Path) -> dict[str, Any]:
    """Score ALS against the popularity baseline on the held-out test split."""
    import numpy as np

    from recsys.als import als_factorize, predict
    from recsys.baseline import recommend_popular
    from recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k
    from recsys.split import train_val_test_split

    cfg = _load_config(config_path)
    df = _load_ratings(cfg)
    scfg, acfg, ecfg = cfg["split"], cfg["als"], cfg["eval"]
    k, thr = ecfg["k"], ecfg["relevant_threshold"]

    train_df, _val, test_df = train_val_test_split(
        df,
        user_col="user",
        val_ratio=scfg["val_ratio"],
        test_ratio=scfg["test_ratio"],
        seed=scfg["seed"],
    )
    R, mask, users, items = _build_matrix(train_df)
    u_index = {u: i for i, u in enumerate(users)}
    i_pos = {it: j for j, it in enumerate(items)}

    U, V = als_factorize(
        R,
        mask,
        rank=acfg["rank"],
        reg=acfg["reg"],
        iters=acfg["iters"],
        seed=acfg["seed"],
    )
    scores = predict(U, V)

    # Per-user relevant sets from the test split.
    relevant: dict[Any, set] = {}
    for u, it, r in zip(
        test_df["user"], test_df["item"], test_df["rating"], strict=False
    ):
        if r >= thr:
            relevant.setdefault(u, set()).add(it)

    als_p, als_r, als_n = [], [], []
    pop_p, pop_r, pop_n = [], [], []
    pop_list = recommend_popular(train_df, k=k, item_col="item")
    for u, rel in relevant.items():
        if u not in u_index:
            continue
        order = np.argsort(-scores[u_index[u]])
        als_rec = [items[j] for j in order[: 10 * k] if items[j] in i_pos][:k]
        grades = dict.fromkeys(rel, 1.0)
        als_p.append(precision_at_k(als_rec, rel, k))
        als_r.append(recall_at_k(als_rec, rel, k))
        als_n.append(ndcg_at_k(als_rec, grades, k))
        pop_p.append(precision_at_k(pop_list, rel, k))
        pop_r.append(recall_at_k(pop_list, rel, k))
        pop_n.append(ndcg_at_k(pop_list, grades, k))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "k": k,
        "als": {
            "precision_at_k": _mean(als_p),
            "recall_at_k": _mean(als_r),
            "ndcg_at_k": _mean(als_n),
        },
        "popularity": {
            "precision_at_k": _mean(pop_p),
            "recall_at_k": _mean(pop_r),
            "ndcg_at_k": _mean(pop_n),
        },
    }
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the seeded synthetic end-to-end demo and print its metrics."""
    from recsys.demo import run_demo

    result = run_demo(seed=seed, out_dir=out_dir)
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    """Build the typer app and dispatch. ``typer`` is imported here only."""
    import typer

    app = typer.Typer(add_completion=False, help=__doc__)
    cfg_opt = typer.Option("config/movielens.yaml", "--config", help="YAML config.")

    @app.command()
    def train_cmd(config: str = cfg_opt) -> None:
        """Fit ALS factors."""
        train(config)

    @app.command(name="recommend")
    def recommend_cmd(config: str = cfg_opt) -> None:
        """Write top-N recommendations."""
        recommend(config)

    @app.command(name="evaluate")
    def evaluate_cmd(config: str = cfg_opt) -> None:
        """Compare ALS against the popularity baseline."""
        evaluate(config)

    @app.command(name="demo")
    def demo_cmd(
        seed: int = typer.Option(0, "--seed", help="Synthesis seed."),
        out_dir: str = typer.Option("outputs", "--out-dir", help="Artefact dir."),
    ) -> None:
        """Run the seeded synthetic end-to-end demo (no config/data needed)."""
        demo(seed=seed, out_dir=out_dir)

    # Register train under the name "train" (the function name carries _cmd).
    app.command(name="train")(train_cmd)

    app(args=argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
