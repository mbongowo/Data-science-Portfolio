"""Known-answer tests for the air-quality alert engine.

Hand-derived behaviour on tiny inputs, no third-party engine.

The cooldown logic is the differentiated piece, so it is tested directly: a
repeat firing of the same (station, rule) within ``cooldown_s`` is suppressed and
counted; a firing once the cooldown has lapsed re-fires; two stations are
independent.
"""

from __future__ import annotations

import pytest

from aqstream import (
    AlertEngine,
    aqi_category_alert,
    spike_alert,
    threshold_alert,
)


def test_threshold_alert_fires_and_not() -> None:
    assert threshold_alert(16.0, 15.0) is True
    assert threshold_alert(15.0, 15.0) is False  # strict >, equality is fine
    assert threshold_alert(14.0, 15.0) is False


def test_threshold_alert_guards_bad_input() -> None:
    with pytest.raises(ValueError):
        threshold_alert(float("nan"), 15.0)
    with pytest.raises(ValueError):
        threshold_alert(10.0, float("inf"))


def test_aqi_category_alert_threshold() -> None:
    assert aqi_category_alert(100) is False  # Moderate
    assert aqi_category_alert(101) is True   # Unhealthy for Sensitive Groups
    assert aqi_category_alert(250) is True   # Very Unhealthy


def test_spike_alert_flags_planted_spike() -> None:
    """A flat baseline then a jump is a spike; steady noise is not."""
    baseline = [10.0, 10.5, 9.8, 10.2, 10.1]
    assert spike_alert(baseline + [40.0], z=3.0) is True
    # Another in-range value is not a spike.
    assert spike_alert(baseline + [10.3], z=3.0) is False


def test_spike_alert_guards() -> None:
    with pytest.raises(ValueError):
        spike_alert([10.0], z=3.0)  # need at least two points
    with pytest.raises(ValueError):
        spike_alert([10.0, 11.0], z=-1.0)


def _engine(cooldown_s):
    """An engine with one threshold rule on pm25 > 15."""
    return AlertEngine(
        rules=[
            {
                "name": "thr",
                "predicate": lambda r: float(r["pm25"]) > 15.0,
                "severity": 2,
                "value_key": "pm25",
            }
        ],
        cooldown_s=cooldown_s,
    )


def test_cooldown_suppresses_repeat_then_refires() -> None:
    """First crossing fires; a repeat inside cooldown is suppressed; later re-fires."""
    eng = _engine(cooldown_s=3600.0)

    fired0 = eng.evaluate({"station": "A", "ts": 0, "pm25": 20.0})
    assert len(fired0) == 1  # first firing

    # 1800 s later, still exceeding, but inside the 3600 s cooldown -> suppressed.
    fired1 = eng.evaluate({"station": "A", "ts": 1800, "pm25": 22.0})
    assert fired1 == []
    assert eng.suppressed == 1

    # Exactly at cooldown boundary (ts == last + cooldown) the cooldown has
    # lapsed, so it re-fires.
    fired2 = eng.evaluate({"station": "A", "ts": 3600, "pm25": 21.0})
    assert len(fired2) == 1
    assert eng.suppressed == 1


def test_no_alert_below_threshold_does_not_touch_cooldown() -> None:
    eng = _engine(cooldown_s=3600.0)
    assert eng.evaluate({"station": "A", "ts": 0, "pm25": 10.0}) == []
    assert eng.suppressed == 0
    # First real crossing still counts as the first firing.
    assert len(eng.evaluate({"station": "A", "ts": 100, "pm25": 20.0})) == 1


def test_multi_station_independence() -> None:
    """Two stations keep separate cooldown state."""
    eng = _engine(cooldown_s=3600.0)
    a = eng.evaluate({"station": "A", "ts": 0, "pm25": 20.0})
    b = eng.evaluate({"station": "B", "ts": 0, "pm25": 20.0})
    assert len(a) == 1 and len(b) == 1  # both fire; independent stations
    assert eng.suppressed == 0


def test_engine_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        AlertEngine(rules=[{"name": "x", "predicate": lambda r: True}], cooldown_s=-1.0)
    with pytest.raises(ValueError):
        AlertEngine(rules=[{"name": "x", "predicate": 42}], cooldown_s=0.0)


def test_engine_requires_station_and_ts() -> None:
    eng = _engine(cooldown_s=0.0)
    with pytest.raises(ValueError):
        eng.evaluate({"ts": 0, "pm25": 20.0})
    with pytest.raises(ValueError):
        eng.evaluate({"station": "A", "pm25": 20.0})


def test_engine_alert_fields() -> None:
    eng = _engine(cooldown_s=0.0)
    (alert,) = eng.evaluate({"station": "A", "ts": 42, "pm25": 20.0})
    assert alert.station == "A"
    assert alert.ts == 42.0
    assert alert.rule == "thr"
    assert alert.value == 20.0
    assert alert.severity == 2
    assert "A" in alert.message
