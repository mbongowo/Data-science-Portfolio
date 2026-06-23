"""Command-line entry point for mlpipe.

Subcommands drive the MLOps loop:

    mlpipe demo                                  # seeded end-to-end demo (no deps)
    mlpipe train   --config config/config.yaml   # train + log to MLflow
    mlpipe serve   --model outputs/model.pkl     # run the FastAPI service
    mlpipe monitor --reference ref.csv --current cur.csv  # drift report

Only ``demo`` runs on the pure-numpy core and needs nothing beyond numpy /
pandas — it is the CI-tested path. ``train`` / ``serve`` / ``monitor`` reach for
the heavy stack (MLflow, uvicorn/FastAPI, Evidently) and import it lazily inside
the command bodies, so importing this module never requires the full stack and
the test suite never pulls it in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the seeded synthetic end-to-end demo and print its metrics."""
    from mlpipe.demo import run_demo

    result = run_demo(seed=seed, out_dir=out_dir)
    print(json.dumps(result, indent=2))
    return result


def train(config_path: str | Path) -> dict[str, Any]:
    """Train the rain-day model on real data and log the run to MLflow.

    Reads the weather frame named in the config, builds features, fits the
    numpy logistic-regression baseline (swap in sklearn here for the heavy path),
    and logs params / metrics / the pickled model through
    :mod:`mlpipe.tracking`. MLflow is imported lazily by that module.
    """
    import pickle

    import pandas as pd
    import yaml

    from mlpipe.features import FEATURE_COLUMNS, make_features, train_test_split_time
    from mlpipe.metrics import accuracy, f1, roc_auc
    from mlpipe.model import LogisticRegression, standardize
    from mlpipe.tracking import log_training, start_run

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    mcfg = cfg["model"]
    df = pd.read_csv(cfg["data"]["weather_csv"])
    feats = make_features(df)
    train_df, test_df = train_test_split_time(feats, frac=cfg["data"]["train_frac"])

    X_train, y_train = train_df[FEATURE_COLUMNS].to_numpy(), train_df["rain_tomorrow"]
    X_test, y_test = test_df[FEATURE_COLUMNS].to_numpy(), test_df["rain_tomorrow"]
    X_train_s, mu, sd = standardize(X_train)
    X_test_s, _, _ = standardize(X_test, mean=mu, std=sd)

    model = LogisticRegression(
        lr=mcfg["lr"], epochs=mcfg["epochs"], l2=mcfg["l2"], seed=mcfg["seed"]
    ).fit(X_train_s, y_train.to_numpy())

    proba = model.predict_proba(X_test_s)
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "test_accuracy": float(accuracy(y_test, preds)),
        "test_f1": float(f1(y_test, preds)),
        "test_roc_auc": float(roc_auc(y_test, proba)),
    }

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    with start_run(run_name="numpy-baseline"):
        log_training(params=dict(mcfg), metrics=metrics, model=model)

    print(json.dumps(metrics, indent=2))
    return metrics


def serve(model_path: str | Path, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI inference service with uvicorn (imported lazily)."""
    import uvicorn

    from mlpipe.serve import create_app

    app = create_app(model_path)
    uvicorn.run(app, host=host, port=port)


def monitor(reference_csv: str | Path, current_csv: str | Path) -> dict[str, Any]:
    """Run a drift report. Uses the pure-numpy core; Evidently is optional.

    Prints the PSI / KS report and its drift summary. To produce the richer
    Evidently HTML dashboard instead, call :func:`mlpipe.monitor.drift_dashboard`.
    """
    import pandas as pd

    from mlpipe.drift import feature_drift_report

    reference = pd.read_csv(reference_csv)
    current = pd.read_csv(current_csv)
    report = feature_drift_report(reference, current)
    print(report.to_string(index=False))
    summary = dict(report.attrs["summary"])
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="mlpipe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Seeded synthetic end-to-end demo.")
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out-dir", default="outputs")

    p_train = sub.add_parser("train", help="Train and log to MLflow.")
    p_train.add_argument("--config", default="config/config.yaml")

    p_serve = sub.add_parser("serve", help="Run the FastAPI inference service.")
    p_serve.add_argument("--model", default="outputs/model.pkl")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    p_mon = sub.add_parser("monitor", help="Run a PSI / KS drift report.")
    p_mon.add_argument("--reference", required=True)
    p_mon.add_argument("--current", required=True)

    args = parser.parse_args(argv)

    if args.command == "demo":
        demo(seed=args.seed, out_dir=args.out_dir)
    elif args.command == "train":
        train(args.config)
    elif args.command == "serve":
        serve(args.model, host=args.host, port=args.port)
    elif args.command == "monitor":
        monitor(args.reference, args.current)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
