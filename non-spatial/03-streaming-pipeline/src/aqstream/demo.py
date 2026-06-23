"""One-command, fully reproducible demo of the pure-Python air-quality core.

This module synthesises a small, **seeded** stream of hourly PM2.5 / PM10
readings for four Cameroon stations over a few days, with a **planted pollution
spike** at one station (a harmattan dust / biomass-smoke episode pushing PM2.5
into the Unhealthy range and holding it there for several hours). It then drives
the *real* core over that stream:

* :func:`aqstream.windows.tumbling_aggregate` for the hourly per-station means,
* :func:`aqstream.aqi.aqi_from_pollutants` / :func:`aqstream.aqi.aqi_category`
  for the AQI and category,
* :class:`aqstream.alerts.AlertEngine` with a threshold rule (WHO PM2.5 24-hour
  guideline) and a spike rule, under a cooldown.

The spike fires alerts; the cooldown suppresses the sustained-exceedance repeats
that would otherwise alert-storm. The metrics are deterministic and pinned by a
test, so the numbers quoted in the README stay honest. Only numpy, pandas and the
standard library are required, so it runs anywhere including CI (no Kafka/Spark).

Run it with ``python -m aqstream.cli demo`` or ``run_demo(seed=0)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aqstream.alerts import AlertEngine, spike_alert
from aqstream.aqi import aqi_category, aqi_from_pollutants
from aqstream.windows import tumbling_aggregate

# Four Cameroon stations (mirrors config/config.yaml). Coordinates are not used
# by the synthetic demo but document which cities the live pipeline would poll.
STATIONS: list[str] = ["Douala", "Yaounde", "Bamenda", "Garoua"]

# Baseline hourly PM2.5 per station (micrograms/m3); urban Douala/Yaounde run
# higher than the others. PM10 is modelled as a multiple of PM2.5.
_BASELINE_PM25: dict[str, float] = {
    "Douala": 22.0,
    "Yaounde": 18.0,
    "Bamenda": 12.0,
    "Garoua": 15.0,
}
_PM10_RATIO: float = 1.8

_DAYS: int = 4
_HOURS: int = _DAYS * 24
_BASE_TS: int = 1_700_000_000  # fixed epoch base so timestamps are stable
_WINDOW_S: float = 3600.0  # 1-hour tumbling windows

# Planted spike: at this station, for this hour span, PM2.5 is pushed high.
_SPIKE_STATION: str = "Garoua"
_SPIKE_START_H: int = 50
_SPIKE_LEN_H: int = 8
_SPIKE_PM25: float = 95.0  # well into the "Unhealthy" AQI band

# Alert configuration (mirrors config/config.yaml).
_WHO_PM25_24H: float = 15.0  # WHO 24-hour guideline, micrograms/m3
_SPIKE_Z: float = 3.0
_COOLDOWN_S: float = 6 * 3600.0  # 6 hours
_SPIKE_LOOKBACK: int = 12  # readings of history the spike rule examines


def _synthesize(seed: int) -> pd.DataFrame:
    """Deterministically synthesise the hourly reading stream.

    Returns a DataFrame with columns ``ts`` (epoch seconds, int), ``station``,
    ``pm25``, ``pm10``, sorted by ``(station, ts)``. One station carries a
    planted multi-hour spike.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, str, float, float]] = []
    for station in STATIONS:
        base = _BASELINE_PM25[station]
        for h in range(_HOURS):
            ts = _BASE_TS + h * 3600
            # Mild diurnal swing plus small gaussian noise around the baseline.
            diurnal = 3.0 * np.sin(2 * np.pi * (h % 24) / 24.0)
            pm25 = base + diurnal + rng.normal(0.0, 1.5)
            if (
                station == _SPIKE_STATION
                and _SPIKE_START_H <= h < _SPIKE_START_H + _SPIKE_LEN_H
            ):
                pm25 = _SPIKE_PM25 + rng.normal(0.0, 2.0)
            pm25 = float(max(0.0, pm25))
            pm10 = float(pm25 * _PM10_RATIO)
            rows.append((ts, station, round(pm25, 2), round(pm10, 2)))

    df = pd.DataFrame(rows, columns=["ts", "station", "pm25", "pm10"])
    df = df.sort_values(["station", "ts"], kind="stable").reset_index(drop=True)
    return df


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the end-to-end demo: synthesise, drive the real core, write files.

    Parameters
    ----------
    seed:
        Seed for numpy's ``default_rng``; fixes the stream so the returned
        metrics are deterministic.
    out_dir:
        Directory for the artifacts (created if missing): ``alerts.csv``,
        ``hourly_aqi.csv`` and ``summary.json``.

    Returns
    -------
    dict
        ``seed``, ``n_readings``, ``n_stations``, ``n_alerts``,
        ``alerts_suppressed_by_cooldown``, ``peak_aqi``, ``peak_station`` and
        ``worst_category``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _synthesize(seed)
    readings = df.to_dict("records")
    n_readings = int(len(df))

    # --- Hourly per-station means via the real tumbling-window core -----------
    pm25_win = tumbling_aggregate(readings, _WINDOW_S, "pm25", agg="mean")
    pm10_win = tumbling_aggregate(readings, _WINDOW_S, "pm10", agg="mean")

    # Build an hourly AQI table keyed by (window_start, station).
    hourly_rows: list[dict] = []
    for key in sorted(pm25_win, key=lambda k: (k[1], k[0])):
        window_start, station = key
        pm25 = pm25_win[key]
        pm10 = pm10_win[key]
        aqi = aqi_from_pollutants(pm25=pm25, pm10=pm10)
        hourly_rows.append(
            {
                "window_start": int(window_start),
                "station": station,
                "pm25": round(pm25, 2),
                "pm10": round(pm10, 2),
                "aqi": int(aqi),
                "category": aqi_category(aqi),
            }
        )
    hourly = pd.DataFrame(hourly_rows)

    # --- Alert engine: threshold (WHO PM2.5) + spike rules, with cooldown -----
    # The spike rule looks back over a per-station rolling buffer of PM2.5.
    history: dict[str, list[float]] = {s: [] for s in STATIONS}

    def threshold_rule(r: dict) -> bool:
        return float(r["pm25"]) > _WHO_PM25_24H

    def spike_rule(r: dict) -> bool:
        # Require a full lookback buffer so the baseline is well estimated; a
        # spike against two near-identical points would otherwise be noise.
        buf = history[r["station"]]
        if len(buf) < _SPIKE_LOOKBACK:
            return False
        window = buf[-_SPIKE_LOOKBACK:] + [float(r["pm25"])]
        return spike_alert(window, z=_SPIKE_Z)

    engine = AlertEngine(
        rules=[
            {
                "name": "who_pm25_24h",
                "predicate": threshold_rule,
                "severity": 2,
                "value_key": "pm25",
                "message": (
                    "{station}: PM2.5 {value} ug/m3 over WHO 24h guideline "
                    "(15) at ts={ts}"
                ),
            },
            {
                "name": "pm25_spike",
                "predicate": spike_rule,
                "severity": 3,
                "value_key": "pm25",
                "message": "{station}: PM2.5 spike to {value} ug/m3 at ts={ts}",
            },
        ],
        cooldown_s=_COOLDOWN_S,
    )

    # Feed readings in time order, interleaved across stations (true stream
    # order), so the cooldown behaves as it would live.
    stream = df.sort_values(["ts", "station"], kind="stable").to_dict("records")
    fired_alerts = []
    for r in stream:
        for alert in engine.evaluate(r):
            fired_alerts.append(alert)
        history[r["station"]].append(float(r["pm25"]))

    # --- Peaks and worst category --------------------------------------------
    peak_idx = int(hourly["aqi"].idxmax())
    peak_aqi = int(hourly.loc[peak_idx, "aqi"])
    peak_station = str(hourly.loc[peak_idx, "station"])
    worst_category = aqi_category(peak_aqi)

    metrics = {
        "seed": int(seed),
        "n_readings": n_readings,
        "n_stations": int(df["station"].nunique()),
        "n_alerts": int(len(fired_alerts)),
        "alerts_suppressed_by_cooldown": int(engine.suppressed),
        "peak_aqi": peak_aqi,
        "peak_station": peak_station,
        "worst_category": worst_category,
    }

    # --- Write artifacts ------------------------------------------------------
    alert_records = [
        {
            "ts": int(a.ts),
            "station": a.station,
            "rule": a.rule,
            "value": round(float(a.value), 2),
            "severity": a.severity,
            "message": a.message,
        }
        for a in fired_alerts
    ]
    alert_cols = ["ts", "station", "rule", "value", "severity", "message"]
    pd.DataFrame(alert_records, columns=alert_cols).to_csv(
        out_path / "alerts.csv", index=False
    )
    hourly.to_csv(out_path / "hourly_aqi.csv", index=False)
    with (out_path / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    return metrics


if __name__ == "__main__":  # pragma: no cover - manual entry point
    print(json.dumps(run_demo(0), default=str, indent=2))
