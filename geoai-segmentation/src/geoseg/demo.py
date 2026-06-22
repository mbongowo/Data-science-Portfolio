"""Runnable, GPU-free demo of the pure-numpy metric core.

``run_demo`` synthesises a small multi-class label map and a deliberately noisy
"prediction" with a controlled error rate, then drives the real metric functions
in :mod:`geoseg.metrics` over them and writes a few artifacts. Nothing here
touches torch or real data: it exists so a reader can clone the repo and see
honest numbers come out of the tested code in well under a second.

The synthetic labels are seeded blobs (one rectangular region per class on a
background of class 0), and the prediction is the truth with a fixed fraction of
pixels flipped to a random wrong class. Both come from a single ``default_rng``,
so a given seed always yields the same numbers.

Run it::

    python -m geoseg.demo            # writes outputs/ and prints the summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geoseg.metrics import (
    cohen_kappa,
    confusion_matrix,
    f1_score,
    frequency_weighted_iou,
    iou_score,
    mean_iou_multiclass,
    per_class_iou,
    pixel_accuracy,
    precision_score,
    recall_score,
)

__all__ = ["run_demo", "main"]


def _synthesize(
    rng: np.random.Generator,
    height: int,
    width: int,
    num_classes: int,
    error_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (truth, prediction) pair of integer label maps.

    Truth is class-0 background with one rectangular blob per non-zero class.
    The prediction copies truth, then flips ``error_rate`` of pixels to a random
    *different* class so the metrics land strictly inside ``(0, 1)``.
    """
    truth = np.zeros((height, width), dtype=np.int64)
    # Lay down one rectangular blob per non-background class.
    for k in range(1, num_classes):
        h0 = rng.integers(0, height - height // 4)
        w0 = rng.integers(0, width - width // 4)
        bh = rng.integers(height // 6, height // 4 + 1)
        bw = rng.integers(width // 6, width // 4 + 1)
        truth[h0 : h0 + bh, w0 : w0 + bw] = k

    pred = truth.copy()
    flip = rng.random((height, width)) < error_rate
    # Replace flipped pixels with a class offset by 1..num_classes-1 (never 0),
    # so a flip always changes the label.
    offset = rng.integers(1, num_classes, size=(height, width))
    pred[flip] = (truth[flip] + offset[flip]) % num_classes
    return truth, pred


def run_demo(seed: int = 0, out_dir: str = "outputs") -> dict:
    """Run the metric demo and write artifacts.

    Parameters
    ----------
    seed : int, optional
        Seed for the synthetic data. Fixed seed -> fixed numbers.
    out_dir : str, optional
        Directory for ``per_class_iou.csv``, ``confusion.csv`` and
        ``summary.json``. Created if missing.

    Returns
    -------
    dict
        ``num_classes``, ``grid_shape``, ``mean_iou``, ``per_class_iou``
        (list, ``nan`` -> ``None``), ``foreground_f1``, ``pixel_accuracy``.
    """
    height, width, num_classes = 64, 64, 4
    error_rate = 0.15
    foreground_class = 1

    rng = np.random.default_rng(seed)
    truth, pred = _synthesize(rng, height, width, num_classes, error_rate)

    # --- drive the real core ------------------------------------------------
    pc_iou = per_class_iou(pred, truth, num_classes)
    m_iou = mean_iou_multiclass(pred, truth, num_classes)
    cm = confusion_matrix(pred, truth, num_classes)
    acc = pixel_accuracy(pred, truth)
    fw_iou = frequency_weighted_iou(pred, truth, num_classes)
    kappa = cohen_kappa(pred, truth, num_classes)

    # Binary foreground scores: class `foreground_class` vs. everything else.
    fg_pred = pred == foreground_class
    fg_truth = truth == foreground_class
    fg_iou = iou_score(fg_pred, fg_truth)
    fg_f1 = f1_score(fg_pred, fg_truth)
    fg_prec = precision_score(fg_pred, fg_truth)
    fg_rec = recall_score(fg_pred, fg_truth)

    # --- artifacts ----------------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pc_lines = ["class,iou"]
    pc_lines += [
        f"{k},{'' if np.isnan(v) else f'{v:.6f}'}" for k, v in enumerate(pc_iou)
    ]
    (out_path / "per_class_iou.csv").write_text("\n".join(pc_lines) + "\n")

    header = "target\\pred," + ",".join(str(k) for k in range(num_classes))
    cm_lines = [header]
    for i in range(num_classes):
        cm_lines.append(str(i) + "," + ",".join(str(int(v)) for v in cm[i]))
    (out_path / "confusion.csv").write_text("\n".join(cm_lines) + "\n")

    per_class_list = [None if np.isnan(v) else float(v) for v in pc_iou]
    summary = {
        "seed": seed,
        "num_classes": num_classes,
        "grid_shape": [height, width],
        "error_rate": error_rate,
        "mean_iou": float(m_iou),
        "per_class_iou": per_class_list,
        "pixel_accuracy": float(acc),
        "frequency_weighted_iou": float(fw_iou),
        "cohen_kappa": float(kappa),
        "foreground_class": foreground_class,
        "foreground_iou": float(fg_iou),
        "foreground_f1": float(fg_f1),
        "foreground_precision": float(fg_prec),
        "foreground_recall": float(fg_rec),
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return {
        "num_classes": num_classes,
        "grid_shape": (height, width),
        "mean_iou": float(m_iou),
        "per_class_iou": per_class_list,
        "foreground_f1": float(fg_f1),
        "pixel_accuracy": float(acc),
    }


def main(argv: list[str] | None = None) -> None:
    """Console entry point for ``python -m geoseg.demo``."""
    parser = argparse.ArgumentParser(description="Run the geoseg metric demo.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic-data seed.")
    parser.add_argument(
        "--out-dir", default="outputs", help="Where to write artifacts."
    )
    args = parser.parse_args(argv)

    result = run_demo(seed=args.seed, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    print(f"\nArtifacts written to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
