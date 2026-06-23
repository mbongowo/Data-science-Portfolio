"""croprec: soil & climate -> best-crop recommendation for Cameroon smallholders.

Given seven agronomic features — soil nutrients N, P, K, plus temperature,
humidity, soil pH and rainfall — the package recommends the best crop with a
ranked top-3 and confidences. The interpretation-critical core is pure
numpy/pandas and has no heavy dependency, so it is always importable and
testable: a stable-softmax multinomial classifier (:mod:`croprec.model`),
classification metrics (:mod:`croprec.metrics`), the data layer
(:mod:`croprec.data`), the recommendation flow (:mod:`croprec.recommend`) and a
seeded synthetic demo (:mod:`croprec.demo`).

Only numpy and pandas are required to import everything re-exported here. The
deployed Streamlit app trains the numpy ``SoftmaxClassifier`` itself at startup
on a bundled sample dataset, so it needs no pre-trained binary. A stronger
scikit-learn RandomForest on the real Kaggle Crop Recommendation dataset is an
optional, documented path in :mod:`croprec.train`, which imports sklearn lazily
and is never imported by the test suite.
"""

from __future__ import annotations

from croprec.data import (
    encode_labels,
    feature_matrix,
    load_crops,
    stratified_split,
)
from croprec.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_f1,
    top_k_accuracy,
)
from croprec.model import SoftmaxClassifier, standardize
from croprec.recommend import recommend

__all__ = [
    "SoftmaxClassifier",
    "standardize",
    "accuracy",
    "confusion_matrix",
    "per_class_f1",
    "macro_f1",
    "top_k_accuracy",
    "load_crops",
    "encode_labels",
    "feature_matrix",
    "stratified_split",
    "recommend",
    "__version__",
]

__version__ = "0.1.0"
