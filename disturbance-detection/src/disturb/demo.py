"""Self-contained, reproducible demo of the disturbance-detection core.

``run_demo`` synthesises a small multi-year NDVI-like series - linear trend +
annual harmonic seasonality + Gaussian noise, with a single *planted* step drop
at a known date - then drives the real pure-numpy core:

* :func:`disturb.decompose.harmonic_decompose` to fit trend + seasonality and
  isolate the residual, and
* :func:`disturb.detect.detect_breakpoint` (CUSUM) on that residual to recover
  the planted disturbance.

It writes three artefacts to ``out_dir`` (``series.csv``, ``components.csv``,
``summary.json``) and returns a dict of the headline metrics. Everything is
deterministic for a given ``seed``, depends only on numpy + stdlib, and runs in
well under a second.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .decompose import harmonic_decompose
from .detect import detect_breakpoint

__all__ = ["run_demo", "synthesize_series"]

# Sampling layout: 16-day composites over 5 years, an annual cycle, and a
# disturbance planted at ~62% through the record.
PERIOD_DAYS = 365.25
STEP_DAYS = 16
N_YEARS = 5
START_DATE = "2018-01-01"
PLANTED_FRACTION = 0.62
PLANTED_DROP = 0.35  # NDVI units removed at the disturbance


def synthesize_series(seed: int = 0) -> dict:
    """Build the synthetic NDVI series and its time axis.

    Returns a dict with ``t`` (days from start), ``times`` (datetime64),
    ``ndvi``, ``planted_index`` and the noise-free truth components.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, N_YEARS * 365, STEP_DAYS, dtype=float)
    times = np.datetime64(START_DATE) + t.astype("timedelta64[D]")
    n = t.size
    planted_index = int(n * PLANTED_FRACTION)

    trend = 0.60 + 0.00003 * t
    seasonal = 0.25 * np.sin(2.0 * np.pi * t / PERIOD_DAYS)
    noise = rng.normal(0.0, 0.02, size=n)
    ndvi = trend + seasonal + noise
    ndvi[planted_index + 1 :] -= PLANTED_DROP  # the planted disturbance

    return {
        "t": t,
        "times": times,
        "ndvi": ndvi,
        "planted_index": planted_index,
        "trend_true": trend,
        "seasonal_true": seasonal,
    }


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the end-to-end demo and write artefacts to ``out_dir``.

    Parameters
    ----------
    seed:
        Seed for the synthetic series (deterministic output).
    out_dir:
        Directory for ``series.csv``, ``components.csv`` and ``summary.json``.
        Created if it does not exist.

    Returns
    -------
    dict
        Keys: ``n_obs``, ``planted_break_index``, ``detected_index``,
        ``detected_magnitude``, ``detected_score``, ``detected``,
        ``seasonal_amplitude``.
    """
    data = synthesize_series(seed)
    t = data["t"]
    times = data["times"]
    ndvi = data["ndvi"]
    planted_index = data["planted_index"]

    # Real core: fit trend + annual seasonality, then scan the residual.
    fit = harmonic_decompose(ndvi, t=t, period=PERIOD_DAYS, n_harmonics=2)
    bp = detect_breakpoint(
        fit.residual, times=times, min_segment=5, threshold=1.0
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # series.csv: the raw synthetic observations.
    with open(out_dir / "series.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "date", "t_days", "ndvi"])
        for k in range(t.size):
            w.writerow([k, str(times[k]), f"{t[k]:.1f}", f"{ndvi[k]:.6f}"])

    # components.csv: the decomposition.
    with open(
        out_dir / "components.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["index", "date", "trend", "seasonal", "residual"])
        for k in range(t.size):
            w.writerow(
                [
                    k,
                    str(times[k]),
                    f"{fit.trend[k]:.6f}",
                    f"{fit.seasonal[k]:.6f}",
                    f"{fit.residual[k]:.6f}",
                ]
            )

    metrics = {
        "n_obs": int(t.size),
        "planted_break_index": int(planted_index),
        "planted_break_date": str(times[planted_index + 1]),
        "detected_index": int(bp.index),
        "detected_date": str(bp.date),
        "detected_magnitude": float(bp.magnitude),
        "detected_score": float(bp.score),
        "detected": bool(bp.detected),
        "seasonal_amplitude": float(fit.seasonal_amplitude(1)),
        "trend_slope_per_year": float(fit.slope * PERIOD_DAYS),
        "seed": int(seed),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # The returned dict is the documented public contract.
    return {
        "n_obs": metrics["n_obs"],
        "planted_break_index": metrics["planted_break_index"],
        "detected_index": metrics["detected_index"],
        "detected_magnitude": metrics["detected_magnitude"],
        "detected_score": metrics["detected_score"],
        "detected": metrics["detected"],
        "seasonal_amplitude": metrics["seasonal_amplitude"],
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print(json.dumps(run_demo(0), indent=2))
