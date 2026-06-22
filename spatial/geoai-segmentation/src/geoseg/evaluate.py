"""Evaluate a trained checkpoint on the held-out test split.

Computes IoU and F1 (via the pure-numpy :mod:`geoseg.metrics`), writes a
``metrics.json``, and saves a qualitative prediction-panel PNG (image | ground
truth | prediction) for a handful of test tiles.

The metric-aggregation helper :func:`aggregate_metrics` is pure-python so it is
unit-testable without torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from geoseg.metrics import f1_score, iou_score

__all__ = ["aggregate_metrics", "save_metrics", "save_prediction_panel", "evaluate"]


def aggregate_metrics(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Mean IoU and F1 over a list of (pred, target) mask pairs."""
    if len(preds) != len(targets):
        raise ValueError("preds and targets must be the same length")
    if not preds:
        return {"iou": 1.0, "f1": 1.0, "n": 0}
    ious = [iou_score(p, t, threshold) for p, t in zip(preds, targets)]
    f1s = [f1_score(p, t, threshold) for p, t in zip(preds, targets)]
    return {
        "iou": float(np.mean(ious)),
        "f1": float(np.mean(f1s)),
        "n": len(preds),
    }


def save_metrics(metrics: dict[str, Any], out_path: str | Path) -> Path:
    """Write a metrics dict to JSON and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_path


def save_prediction_panel(
    images: list[np.ndarray],
    targets: list[np.ndarray],
    preds: list[np.ndarray],
    out_path: str | Path,
    max_rows: int = 4,
) -> Path:  # pragma: no cover - needs matplotlib
    """Save an image|GT|prediction panel PNG for a few samples."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = min(len(images), max_rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        img = np.transpose(images[i], (1, 2, 0))
        img = (img - img.min()) / (img.ptp() + 1e-6)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(np.squeeze(targets[i]), cmap="gray")
        axes[i, 1].set_title("ground truth")
        axes[i, 2].imshow(np.squeeze(preds[i]), cmap="gray")
        axes[i, 2].set_title("prediction")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def evaluate(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path = "outputs/eval",
    tile_size: int = 256,
    threshold: float = 0.5,
) -> dict[str, float]:  # pragma: no cover - needs torch
    """Run inference over the test split and write metrics + panel."""
    import torch  # noqa: PLC0415

    from geoseg.datamodule import GeoSegDataModule
    from geoseg.model import SegmentationModule

    output_dir = Path(output_dir)
    model = SegmentationModule.load_from_checkpoint(str(checkpoint))
    model.eval()

    dm = GeoSegDataModule(data_dir=str(data_dir), tile_size=tile_size)
    dm.setup("test")
    loader = dm.test_dataloader()

    images, targets, preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"])
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()
            for j in range(batch["image"].shape[0]):
                images.append(batch["image"][j].cpu().numpy())
                targets.append(batch["mask"][j].cpu().numpy())
                preds.append(pred[j].cpu().numpy())

    metrics = aggregate_metrics(preds, targets, threshold)
    save_metrics(metrics, output_dir / "metrics.json")
    if images:
        save_prediction_panel(images, targets, preds, output_dir / "panel.png")
    return metrics
