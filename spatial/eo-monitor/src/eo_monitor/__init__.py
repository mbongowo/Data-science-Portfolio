"""eo-monitor: config-driven Sentinel-2 vegetation/moisture anomaly monitoring.

The pure-numpy index and anomaly helpers are re-exported here for convenience.
Heavy geospatial modules (catalog, cube, io) are deliberately *not* imported at
package import time so the math stays usable with only numpy installed.
"""

from __future__ import annotations

from eo_monitor.anomaly import (
    anomaly_cube,
    anomaly_fraction,
    baseline_statistics,
    classify_anomaly,
    robust_zscore,
    zscore_anomaly,
)
from eo_monitor.indices import (
    compute_index,
    evi2,
    nbr,
    ndmi,
    ndvi,
    ndwi,
    normalized_difference,
    required_bands,
    savi,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # indices
    "ndvi",
    "ndwi",
    "ndmi",
    "savi",
    "evi2",
    "nbr",
    "normalized_difference",
    "compute_index",
    "required_bands",
    # anomaly
    "baseline_statistics",
    "zscore_anomaly",
    "anomaly_cube",
    "robust_zscore",
    "anomaly_fraction",
    "classify_anomaly",
]
