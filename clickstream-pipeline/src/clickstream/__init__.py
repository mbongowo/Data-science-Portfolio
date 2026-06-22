"""clickstream-pipeline: event-time stream processing, with a pure-Python core.

This package implements the windowing, sessionisation, funnel, and watermark
logic of a clickstream pipeline as plain Python/pandas functions, plus a guarded
streaming layer (Kafka + Spark Structured Streaming) that runs the same logic at
scale.

The package is split so that the interpretation-critical numeric core has no
heavy dependency and is always importable and testable. Only numpy, pandas, and
pyyaml are needed to import everything re-exported here. The streaming engine in
:mod:`clickstream.pipeline` imports ``pyspark`` and ``kafka`` lazily *inside*
its functions, so neither this module nor the test suite requires them.
"""

from __future__ import annotations

from clickstream.aggregate import events_per_minute
from clickstream.streaming import (
    funnel_time_to_convert,
    reorder_within_lateness,
    retention,
    top_k_heavy_hitters,
)
from clickstream.watermark import advance_watermark, is_late
from clickstream.windows import (
    funnel,
    sessionize,
    sliding_counts,
    tumbling_counts,
)

__all__ = [
    "tumbling_counts",
    "sliding_counts",
    "sessionize",
    "funnel",
    "is_late",
    "advance_watermark",
    "events_per_minute",
    "top_k_heavy_hitters",
    "reorder_within_lateness",
    "funnel_time_to_convert",
    "retention",
    "__version__",
]

__version__ = "0.1.0"
