"""geoseg: reproducible deep-learning semantic segmentation for Earth-observation imagery.

The package is designed so that *importing it never pulls in heavy optional
dependencies* (torch, lightning, rasterio, ...). Pure-python / numpy utilities
(metrics, deterministic split logic) are always importable, which keeps the test
suite runnable on a bare CI machine.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Pure-numpy public surface. These modules import only numpy/stdlib, so
# re-exporting them here keeps `import geoseg` torch-free and CI-installable.
from geoseg.metrics import (
    cohen_kappa,
    confusion_counts,
    confusion_matrix,
    f1_score,
    frequency_weighted_iou,
    iou_score,
    mean_iou,
    mean_iou_multiclass,
    per_class_iou,
    per_class_precision,
    per_class_recall,
    pixel_accuracy,
    precision_score,
    recall_score,
)
from geoseg.tiling import stitch, tile_indices

__all__ = [
    "__version__",
    # metrics
    "iou_score",
    "f1_score",
    "precision_score",
    "recall_score",
    "confusion_counts",
    "mean_iou",
    "per_class_iou",
    "mean_iou_multiclass",
    "confusion_matrix",
    "pixel_accuracy",
    "frequency_weighted_iou",
    "cohen_kappa",
    "per_class_precision",
    "per_class_recall",
    # tiling
    "tile_indices",
    "stitch",
]
