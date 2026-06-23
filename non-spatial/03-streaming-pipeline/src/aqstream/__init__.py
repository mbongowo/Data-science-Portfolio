"""aqstream: real-time air-quality streaming and alerting for Cameroon cities.

This package turns a stream of air-quality readings (PM2.5 / PM10) into US-EPA
AQI sub-indices, aggregates them over tumbling time windows, and runs an alert
engine that fires when pollution crosses WHO/EPA thresholds or spikes, with a
per-(station, rule) cooldown so a sustained exceedance does not alert-storm.

The package is split so that the interpretation-critical core has no heavy
dependency and is always importable and testable. Only numpy, pandas, and
(optionally) pyyaml are needed to import everything re-exported here. The
ingestion (:mod:`aqstream.ingest`), streaming (:mod:`aqstream.stream`), storage
(:mod:`aqstream.sink`) and notification (:mod:`aqstream.notify`) layers import
``requests`` / ``kafka`` / ``pyspark`` / ``duckdb`` lazily *inside* their
functions, so neither this module nor the test suite requires them. Those layers
run in the docker-compose stack (or the opt-in Azure path), not in CI.
"""

from __future__ import annotations

from aqstream.alerts import (
    Alert,
    AlertEngine,
    aqi_category_alert,
    spike_alert,
    threshold_alert,
)
from aqstream.aqi import (
    aqi_category,
    aqi_from_pollutants,
    pm10_to_aqi,
    pm25_to_aqi,
)
from aqstream.windows import dedupe, rolling_mean, tumbling_aggregate

__all__ = [
    "pm25_to_aqi",
    "pm10_to_aqi",
    "aqi_from_pollutants",
    "aqi_category",
    "tumbling_aggregate",
    "rolling_mean",
    "dedupe",
    "threshold_alert",
    "aqi_category_alert",
    "spike_alert",
    "Alert",
    "AlertEngine",
    "__version__",
]

__version__ = "0.1.0"
