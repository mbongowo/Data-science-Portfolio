"""Air-quality alert engine (pure Python) — the differentiated core.

This module decides *when to fire an alert* on a stream of air-quality readings
and, crucially, *when to stay quiet*. It has no third-party dependency beyond the
standard library, so it is always importable and is covered by hand-derived
known-answer tests.

Three alert rules are provided:

* :func:`threshold_alert` — a value crosses a fixed threshold (e.g. PM2.5 above
  the WHO 24-hour guideline of 15 micrograms/m3).
* :func:`aqi_category_alert` — the AQI category reaches "Unhealthy for Sensitive
  Groups" or worse.
* :func:`spike_alert` — a reading sits ``z`` standard deviations above the
  trailing rolling mean (a sudden pollution spike, e.g. a harmattan dust surge or
  a biomass-burning episode), even if the absolute level is not yet extreme.

The :class:`AlertEngine` applies a set of rules per station and **suppresses
repeat alerts** of the same ``(station, rule)`` within ``cooldown_s`` seconds.
This debounce is what stops a sustained exceedance — which would otherwise fire
on every reading — from turning into an alert-storm: the first crossing fires,
the rest are counted as suppressed until the cooldown lapses.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Ordered AQI categories, worst last; index gives an ordinal severity.
_CATEGORY_ORDER: list[str] = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

# Category at/above which aqi_category_alert fires.
_ALERT_FROM_CATEGORY = "Unhealthy for Sensitive Groups"


@dataclass(frozen=True)
class Alert:
    """A fired alert.

    Attributes
    ----------
    station:
        Station identifier the alert is about.
    ts:
        Epoch-second timestamp of the reading that triggered it.
    rule:
        Name of the rule that fired (used together with ``station`` for the
        cooldown key).
    value:
        The numeric value that triggered the alert (e.g. the PM2.5 or the AQI).
    severity:
        Ordinal severity, larger is worse.
    message:
        Human-readable description.
    """

    station: str
    ts: float
    rule: str
    value: float
    severity: int
    message: str


def threshold_alert(
    value: float, threshold: float, level: str = "warning"
) -> bool:
    """Return ``True`` when ``value`` is strictly above ``threshold``.

    A reading exactly equal to the threshold does not alert (strict ``>``), so a
    guideline value itself is treated as the last acceptable level.

    Raises
    ------
    ValueError
        If ``value`` or ``threshold`` is not a finite number.
    """
    _require_finite(value, "value")
    _require_finite(threshold, "threshold")
    _ = level  # carried for the caller's labelling; not used in the comparison
    return float(value) > float(threshold)


def aqi_category_alert(aqi: float) -> bool:
    """Return ``True`` when the AQI category is "Unhealthy for Sensitive Groups"+.

    Imports :func:`aqstream.aqi.aqi_category` locally to keep this module's
    import graph free of any cycle; both are pure standard-library code.

    Raises
    ------
    ValueError
        If ``aqi`` is not a finite, non-negative number.
    """
    _require_finite(aqi, "aqi")
    if aqi < 0:
        raise ValueError("aqi must be non-negative.")

    from aqstream.aqi import aqi_category

    category = aqi_category(aqi)
    return _CATEGORY_ORDER.index(category) >= _CATEGORY_ORDER.index(
        _ALERT_FROM_CATEGORY
    )


def spike_alert(series: Sequence[float], z: float = 3.0) -> bool:
    """Flag the *last* value as a spike if it is ``z`` std above the prior mean.

    The baseline is the mean and population standard deviation of all values
    *before* the last one. The final value is a spike when::

        last > baseline_mean + z * baseline_std

    A flat or near-flat baseline (zero std) never flags on noise; it flags only a
    strictly higher last value when ``z == 0``. ``series`` needs at least two
    points (one baseline value plus the candidate).

    Raises
    ------
    ValueError
        If ``series`` has fewer than two values or ``z`` is negative.
    """
    if z < 0:
        raise ValueError("z must be non-negative.")
    vals = [float(v) for v in series]
    if len(vals) < 2:
        raise ValueError("series must have at least two values.")

    baseline = vals[:-1]
    last = vals[-1]
    mean = sum(baseline) / len(baseline)
    var = sum((v - mean) ** 2 for v in baseline) / len(baseline)
    std = math.sqrt(var)
    return last > mean + z * std


@dataclass
class _Rule:
    """An engine rule: a name, a predicate over a reading, and severity/message."""

    name: str
    predicate: Callable[[dict], bool]
    severity: int
    value_key: str
    message: str


@dataclass
class AlertEngine:
    """Evaluate alert rules per station with a per-(station, rule) cooldown.

    Parameters
    ----------
    rules:
        List of rule specs. Each spec is a dict with keys:

        * ``name`` — rule name (str).
        * ``predicate`` — callable ``reading -> bool``; ``True`` means fire.
        * ``severity`` — ordinal severity (int); larger is worse. Default 1.
        * ``value_key`` — reading field to report as the alert ``value``.
          Default ``"value"``.
        * ``message`` — template; ``{station}``/``{value}``/``{ts}`` are filled.
          Default a generic message.

    cooldown_s:
        Suppression window in seconds. After a ``(station, rule)`` fires at time
        ``t``, further firings of that same pair with ``ts < t + cooldown_s`` are
        suppressed (counted in :attr:`suppressed`) rather than emitted. A firing
        at exactly ``t + cooldown_s`` is allowed (the cooldown has lapsed).

    The engine is stateful: it remembers the last fire time per ``(station,
    rule)`` across calls, so :meth:`evaluate` must be fed readings in
    non-decreasing time order per station.
    """

    rules: list[dict]
    cooldown_s: float
    _rules: list[_Rule] = field(init=False, default_factory=list)
    _last_fired: dict[tuple[str, str], float] = field(
        init=False, default_factory=dict
    )
    suppressed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be non-negative.")
        for spec in self.rules:
            if "name" not in spec or "predicate" not in spec:
                raise ValueError("each rule needs at least 'name' and 'predicate'.")
            if not callable(spec["predicate"]):
                raise ValueError(
                    f"predicate for rule {spec['name']!r} is not callable."
                )
            self._rules.append(
                _Rule(
                    name=str(spec["name"]),
                    predicate=spec["predicate"],
                    severity=int(spec.get("severity", 1)),
                    value_key=str(spec.get("value_key", "value")),
                    message=str(
                        spec.get(
                            "message",
                            "{station}: {rule} fired (value={value}) at {ts}",
                        )
                    ),
                )
            )

    def evaluate(self, reading: dict) -> list[Alert]:
        """Evaluate all rules against one reading, honouring the cooldown.

        Returns the list of :class:`Alert` objects that *fired* (were not
        suppressed). Suppressed firings increment :attr:`suppressed`.

        Raises
        ------
        ValueError
            If ``reading`` lacks ``station`` or ``ts``.
        """
        for field_name in ("station", "ts"):
            if field_name not in reading:
                raise ValueError(f"reading is missing required field {field_name!r}.")

        station = str(reading["station"])
        ts = float(reading["ts"])
        fired: list[Alert] = []

        for rule in self._rules:
            try:
                hit = bool(rule.predicate(reading))
            except (KeyError, TypeError, ValueError):
                # A rule that cannot be evaluated on this reading simply does not
                # fire; a malformed reading must not crash the stream.
                hit = False
            if not hit:
                continue

            key = (station, rule.name)
            last = self._last_fired.get(key)
            if last is not None and ts < last + self.cooldown_s:
                self.suppressed += 1
                continue

            self._last_fired[key] = ts
            value = float(reading.get(rule.value_key, float("nan")))
            message = rule.message.format(
                station=station, rule=rule.name, value=value, ts=ts
            )
            fired.append(
                Alert(
                    station=station,
                    ts=ts,
                    rule=rule.name,
                    value=value,
                    severity=rule.severity,
                    message=message,
                )
            )
        return fired


def _require_finite(x: float, name: str) -> None:
    """Raise ``ValueError`` unless ``x`` is a finite real number."""
    try:
        xf = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if math.isnan(xf) or math.isinf(xf):
        raise ValueError(f"{name} must be finite.")
