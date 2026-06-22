"""eets: multi-year vegetation-change & forest-loss detection from S2 time series.

This package turns a stack of Sentinel-2 spectral-index images over several
years into *change numbers*: per-period index time series, cloud-robust
temporal composites (baseline vs recent), a per-pixel change map, a
loss/stable/gain classification, hectares of loss and gain, and burn severity
(dNBR) classes. That quantification — the index series, the composites, the
change map, the hectares — is the runnable, tested contribution.

The numeric core (:mod:`eets.indices`, :mod:`eets.timeseries`,
:mod:`eets.change`) and the demo (:mod:`eets.demo`) depend only on numpy and the
standard library, so they import and test anywhere. The default real-data path
(:mod:`eets.stac`, Earth Search Sentinel-2 L2A, no auth) and the optional Earth
Engine path (:mod:`eets.gee`) import their heavy dependencies lazily, inside
functions, and are not imported here or by the test suite.
"""

from __future__ import annotations

from eets.change import (
    change_map,
    change_stats,
    classify_burn_severity,
    classify_change,
    dnbr,
    severity_stats,
)
from eets.demo import run_demo
from eets.indices import nbr, ndvi, ndwi, normalized_difference
from eets.timeseries import index_timeseries, mask_invalid, temporal_composite

__all__ = [
    "normalized_difference",
    "ndvi",
    "ndwi",
    "nbr",
    "index_timeseries",
    "temporal_composite",
    "mask_invalid",
    "change_map",
    "classify_change",
    "change_stats",
    "dnbr",
    "classify_burn_severity",
    "severity_stats",
    "run_demo",
    "__version__",
]

__version__ = "0.1.0"
