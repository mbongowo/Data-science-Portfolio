"""lcnet: land-cover patch classification, set up to reproduce.

The package is built so that *importing it never pulls in heavy optional
dependencies* (torch, torchvision, torchgeo, rasterio, ...). The pure-numpy
core — classification metrics, a trainable softmax baseline, and the
imagery-to-feature bridge — is always importable, which keeps the test suite
runnable on a bare CI machine without a GPU.

The real TorchGeo ResNet fine-tune on EuroSAT lives behind lazy imports in
:mod:`lcnet.train` (needs a GPU/Colab) and is not imported here.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Pure-numpy public surface. These modules import only numpy/stdlib, so
# re-exporting them here keeps `import lcnet` torch-free and CI-installable.
from lcnet.classifier import SoftmaxClassifier, standardize
from lcnet.data import band_stats, patch_features, stratified_split
from lcnet.metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    micro_f1,
    per_class_f1,
    per_class_precision,
    per_class_recall,
    top_k_accuracy,
)

__all__ = [
    "__version__",
    # metrics
    "confusion_matrix",
    "accuracy",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "macro_f1",
    "micro_f1",
    "cohen_kappa",
    "top_k_accuracy",
    # classifier
    "SoftmaxClassifier",
    "standardize",
    # data
    "stratified_split",
    "band_stats",
    "patch_features",
]
