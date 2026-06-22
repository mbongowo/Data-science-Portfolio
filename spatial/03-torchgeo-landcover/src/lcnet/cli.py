"""Command-line entry point for lcnet (land-cover classification).

Three subcommands:

* ``demo``    — dependency-free synthetic run; needs only numpy. Synthesises an
                EuroSAT-like dataset, trains the from-scratch softmax baseline,
                and writes ``outputs/{confusion_matrix.csv, metrics.json}``.
* ``train``   — fine-tune a ResNet18 on EuroSAT via TorchGeo (heavy; GPU/Colab).
* ``compare`` — run pretrained vs from-scratch and print both metric sets (heavy).

Only ``demo`` runs without the deep-learning stack. The torch / torchgeo imports
for ``train`` and ``compare`` happen inside :mod:`lcnet.train`, which is imported
lazily in the handlers below, so importing this module — for ``--help`` or in a
test — never requires those dependencies or a GPU.
"""

from __future__ import annotations

import argparse
import json


def _run_train(args: argparse.Namespace) -> int:
    from lcnet.train import train_eurosat  # noqa: PLC0415

    metrics = train_eurosat(
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_root=args.data_root,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    from lcnet.train import compare_transfer  # noqa: PLC0415

    results = compare_transfer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_root=args.data_root,
    )
    print(json.dumps(results, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lcnet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo", help="Dependency-free synthetic classification run (numpy)."
    )
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out-dir", default="outputs")

    p_train = sub.add_parser(
        "train", help="Fine-tune a ResNet18 on EuroSAT via TorchGeo (GPU/Colab)."
    )
    p_train.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start from ImageNet weights (--no-pretrained for from-scratch).",
    )
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--data-root", default="data/eurosat")

    p_cmp = sub.add_parser(
        "compare", help="Pretrained vs from-scratch EuroSAT fine-tune (GPU/Colab)."
    )
    p_cmp.add_argument("--epochs", type=int, default=10)
    p_cmp.add_argument("--batch-size", type=int, default=64)
    p_cmp.add_argument("--lr", type=float, default=1e-3)
    p_cmp.add_argument("--seed", type=int, default=0)
    p_cmp.add_argument("--data-root", default="data/eurosat")

    args = parser.parse_args(argv)

    if args.command == "demo":
        from lcnet.demo import run_demo  # noqa: PLC0415

        summary = run_demo(seed=args.seed, out_dir=args.out_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "train":
        return _run_train(args)

    if args.command == "compare":
        return _run_compare(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
