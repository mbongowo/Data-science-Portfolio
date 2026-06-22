"""log-anomaly: unsupervised anomaly detection over system logs.

The pipeline collapses raw log lines to event templates (Drain-lite masking),
counts templates per session to form an event-count matrix, scores each session
with a transparent unsupervised detector (PCA reconstruction error or a z-score
rule), and — when labels exist (Loghub HDFS_v1) — measures precision / recall /
F1.

The interpretation-critical numeric core (templating, features, detectors,
metrics) is pure numpy / pandas / stdlib with no heavy dependency, so it is
always importable and is covered by hand-derived known-answer tests. The
scale-out parts (Spark ingest/parse, optional sklearn IsolationForest) live in
:mod:`loganomaly.spark_pipeline` behind lazy imports and are not pulled in here.
"""

from __future__ import annotations

from loganomaly.detect import (
    flag,
    pca_reconstruction_error,
    zscore_anomalies,
)
from loganomaly.evaluate import confusion_matrix, precision_recall_f1
from loganomaly.features import event_count_matrix
from loganomaly.templating import mask_line, template_id

__all__ = [
    "mask_line",
    "template_id",
    "event_count_matrix",
    "pca_reconstruction_error",
    "zscore_anomalies",
    "flag",
    "precision_recall_f1",
    "confusion_matrix",
    "__version__",
]

__version__ = "0.1.0"
