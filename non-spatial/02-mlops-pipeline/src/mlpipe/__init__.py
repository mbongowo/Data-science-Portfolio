"""mlpipe: rain-day prediction from notebook to a monitored service.

This package takes a small weather model the whole way round the MLOps loop:
build features from a daily weather frame, train a model, track the run, serve
predictions, and then *watch for drift* once the model is live.

The package is split so the interpretation-critical numeric core has no
third-party dependency beyond numpy / pandas (and pyyaml for config). Those core
modules are always importable and fully unit-tested:

* :mod:`mlpipe.drift`    — PSI / KS drift detection (the differentiated star);
* :mod:`mlpipe.metrics`  — classification and regression metrics;
* :mod:`mlpipe.model`    — a pure-numpy logistic-regression classifier;
* :mod:`mlpipe.features` — pandas feature engineering with a no-leakage split.

The MLOps plumbing lives behind lazy wrappers that are *not* imported here and
never pulled in by the test suite: :mod:`mlpipe.tracking` (MLflow),
:mod:`mlpipe.serve` (FastAPI), and :mod:`mlpipe.monitor` (Evidently). Each has a
free-local path and an opt-in Azure ML path. Importing this package therefore
costs nothing beyond numpy / pandas.
"""

from __future__ import annotations

from mlpipe.drift import feature_drift_report, ks_statistic, psi
from mlpipe.features import make_features, train_test_split_time
from mlpipe.metrics import (
    accuracy,
    confusion_counts,
    f1,
    mae,
    precision,
    r2,
    recall,
    rmse,
    roc_auc,
)
from mlpipe.model import LogisticRegression, standardize

__all__ = [
    "psi",
    "ks_statistic",
    "feature_drift_report",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_counts",
    "roc_auc",
    "rmse",
    "mae",
    "r2",
    "LogisticRegression",
    "standardize",
    "make_features",
    "train_test_split_time",
    "__version__",
]

__version__ = "0.1.0"
