"""Command-line interface for the crop recommender.

The ``demo`` command runs the pure-numpy core (synthesise -> standardise ->
softmax classifier -> evaluate) and is the CI-tested, runnable contribution. The
``train`` command is a thin stub over the optional scikit-learn RandomForest in
:mod:`croprec.train`; its heavy import lives inside that module, so importing
this CLI is cheap and the tested core stays free of optional dependencies.

Run ``python -m croprec.cli demo`` to reproduce the result numbers.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the seeded synthetic demo end-to-end and print the summary."""
    from croprec.demo import run_demo

    metrics = run_demo(seed=args.seed, out_dir=args.out_dir)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the optional sklearn RandomForest on the real Kaggle CSV (lazy)."""
    from croprec.train import train_random_forest

    metrics = train_random_forest(args.csv, model_out=args.model_out, seed=args.seed)
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI."""
    parser = argparse.ArgumentParser(
        prog="croprec",
        description="Crop recommender: soil & climate -> best crop (Cameroon).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the seeded synthetic demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out-dir", default="outputs")
    demo.set_defaults(func=_cmd_demo)

    train = sub.add_parser(
        "train",
        help="train the optional sklearn RandomForest on the real Kaggle CSV",
    )
    train.add_argument("--csv", required=True, help="path to Crop_recommendation.csv")
    train.add_argument("--model-out", default="models/crop_rf.joblib")
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``croprec``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
