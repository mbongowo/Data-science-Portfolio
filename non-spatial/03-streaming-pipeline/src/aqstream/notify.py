"""Notification layer: deliver fired alerts to a webhook / Slack / log (guarded).

This module runs in the docker-compose stack, not in CI. The only network call
(``send_webhook``) imports ``requests`` lazily, so importing this module is free
and the test suite never needs it. It consumes the :class:`aqstream.alerts.Alert`
objects the pure alert engine emits.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("aqstream.notify")


def alert_payload(alert: Any) -> dict:
    """Build a JSON-serialisable payload from an :class:`Alert` (pure)."""
    return {
        "station": alert.station,
        "ts": int(alert.ts),
        "rule": alert.rule,
        "value": float(alert.value),
        "severity": int(alert.severity),
        "message": alert.message,
    }


def send_webhook(alert: Any, url: str, *, timeout: float = 10.0) -> int:
    """POST an alert to a generic JSON webhook (lazy ``requests`` import).

    Returns the HTTP status code. Works for any endpoint that accepts a JSON
    body, including Slack incoming webhooks (which read the ``text`` field).
    """
    import requests  # lazy: only needed when actually notifying

    payload = alert_payload(alert)
    payload["text"] = alert.message  # Slack-friendly field
    resp = requests.post(url, json=payload, timeout=timeout)
    return resp.status_code


def log_alert(alert: Any) -> None:
    """Log a fired alert at WARNING (the no-dependency default notifier)."""
    logger.warning(
        "ALERT [%s] %s: %s", alert.severity, alert.station, alert.message
    )
