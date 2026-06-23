"""A one-command, dependency-light demo of the weather pipeline core.

This drives the **real** pure-pandas core (:mod:`weatherpipe.ingest`,
:mod:`weatherpipe.validate`, :mod:`weatherpipe.transform`) over a small,
deterministic, synthetic year of daily weather for the five Cameroon stations. It
needs only numpy / pandas / pyyaml + stdlib — no requests, dbt, duckdb, prefect
or streamlit — so it reproduces real, committed metrics anywhere, including CI.

What it does:

1. Synthesizes ~1 year of daily weather per station with
   ``numpy.random.default_rng(seed)``: a seasonal temperature cycle plus a
   wet-season precipitation pattern tuned per city (Maroua/Garoua drier, far
   north; Douala wetter, coastal).
2. Plants a few bad rows on purpose (``tmin_c > tmax_c``, a negative
   precipitation, and a duplicate ``(station, date)``) so the validator has
   something to reject.
3. Runs the real ``normalize -> validate -> monthly_summary`` path and writes
   ``outputs/monthly_summary.csv`` and ``outputs/validation_report.json``.

It returns a small dict of real numbers (record / station / rejection counts, the
hottest station-month and the wettest month) that ``tests/test_demo.py`` pins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weatherpipe.ingest import normalize
from weatherpipe.transform import monthly_summary
from weatherpipe.validate import validate_weather

# Five Cameroon stations: (name, baseline mean temp C, seasonal amplitude C,
# annual precip scale mm/day in the wet season). The far-north Sahel stations
# (Maroua, Garoua) are hotter and drier; coastal Douala is wet; the western
# highlands (Bamenda) are cooler.
_STATIONS = (
    ("Douala", 27.0, 2.5, 12.0),
    ("Yaounde", 24.0, 3.0, 7.0),
    ("Maroua", 29.0, 6.0, 3.0),
    ("Garoua", 29.5, 6.5, 3.5),
    ("Bamenda", 21.0, 3.5, 9.0),
)

# A full, non-leap year of daily data.
_START = "2023-01-01"
_END = "2023-12-31"

#: Number of bad rows the demo plants (one of each kind, see ``_plant_bad_rows``).
N_PLANTED_BAD = 3


def synthesize_station(
    rng: np.random.Generator,
    name: str,
    base_temp: float,
    amp: float,
    wet_scale: float,
    dates: pd.DatetimeIndex,
) -> list[dict]:
    """Synthesize one clean daily record per date for a station."""
    n = len(dates)
    doy = dates.dayofyear.to_numpy().astype(float)
    # Seasonal temperature: warmest around day 90 (boreal spring in the tropics).
    season = amp * np.sin(2 * np.pi * (doy - 90) / 365.0)
    tmean = base_temp + season + rng.normal(0.0, 1.0, size=n)
    spread = 4.0 + rng.uniform(0.0, 3.0, size=n)
    tmin = tmean - spread / 2.0
    tmax = tmean + spread / 2.0

    # Wet season roughly April-October (a raised cosine peaking mid-year),
    # gamma-distributed daily totals scaled by the city's wetness.
    wet = np.clip(np.sin(np.pi * (doy - 60) / 220.0), 0.0, None)
    rain_prob = 0.15 + 0.7 * wet
    is_wet = rng.uniform(size=n) < rain_prob
    amounts = rng.gamma(shape=2.0, scale=wet_scale, size=n) * wet
    precip = np.where(is_wet, amounts, 0.0)

    records = []
    for i in range(n):
        records.append(
            {
                "date": dates[i].strftime("%Y-%m-%d"),
                "tmin_c": round(float(tmin[i]), 2),
                "tmax_c": round(float(tmax[i]), 2),
                "tmean_c": round(float(tmean[i]), 2),
                "precip_mm": round(float(precip[i]), 2),
            }
        )
    return records


def _plant_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Plant exactly ``N_PLANTED_BAD`` bad rows for the validator to reject.

    One inverted temperature pair (``tmin_c > tmax_c``), one negative
    precipitation, and one duplicated ``(station, date)`` row.
    """
    out = df.copy().reset_index(drop=True)
    # 1. tmin > tmax on the first row.
    out.loc[0, "tmin_c"] = 40.0
    out.loc[0, "tmax_c"] = 10.0
    # 2. negative precipitation on the second row.
    out.loc[1, "precip_mm"] = -5.0
    # 3. duplicate (station, date): copy an existing row verbatim.
    dup = out.iloc[10].copy()
    out = pd.concat([out, dup.to_frame().T], ignore_index=True)
    return out


def run_demo(seed: int = 0, out_dir: str = "outputs") -> dict:
    """Run the weather pipeline core end to end on synthetic data.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng``; fixes every draw, so the returned
        metrics and written artifacts are fully reproducible.
    out_dir:
        Directory for the artifacts (created if missing).

    Returns
    -------
    dict
        ``n_records``, ``n_stations``, ``n_rejected``, ``pct_valid``,
        ``hottest_station_month`` (``station`` / ``year`` / ``month`` / ``tmean``)
        and ``wettest_month_total_precip_mm``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(_START, _END, freq="D")

    frames = []
    for name, base_temp, amp, wet_scale, dates_in in (
        (n, b, a, w, dates) for (n, b, a, w) in _STATIONS
    ):
        records = synthesize_station(rng, name, base_temp, amp, wet_scale, dates_in)
        frames.append(normalize(records, name))
    clean_input = pd.concat(frames, ignore_index=True)

    # Plant defects, then run the real validate -> transform path.
    dirty = _plant_bad_rows(clean_input)
    clean, report = validate_weather(dirty)
    monthly = monthly_summary(clean)

    # Headline aggregates from the clean monthly mart.
    hottest = monthly.loc[monthly["tmean_mean"].idxmax()]
    wettest_total = float(monthly["precip_total_mm"].max())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out / "monthly_summary.csv", index=False)
    (out / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    return {
        "seed": seed,
        "n_records": int(len(clean)),
        "n_stations": int(clean["station"].nunique()),
        "n_rejected": int(report["n_rejected"]),
        "pct_valid": float(report["pct_valid"]),
        "hottest_station_month": {
            "station": str(hottest["station"]),
            "year": int(hottest["year"]),
            "month": int(hottest["month"]),
            "tmean": round(float(hottest["tmean_mean"]), 2),
        },
        "wettest_month_total_precip_mm": round(wettest_total, 2),
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
