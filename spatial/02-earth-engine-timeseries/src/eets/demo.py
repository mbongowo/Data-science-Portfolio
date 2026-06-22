"""A self-contained, dependency-free demo of the time-series + change core.

This module runs the whole project end to end without Earth Engine, without the
STAC stack, without a network call, and without any downloaded imagery. It
deterministically synthesises a small multi-year stack of Sentinel-2-like NDVI
scenes over a vegetated block: NDVI is high everywhere in the baseline years,
then a rectangular **planted clearing** appears in the recent years (NDVI drops
there), with mild per-scene noise and a few synthetic "cloud" pixels masked out
through a scene-classification layer (SCL).

It then drives the *real* pure-numpy core: the spatial-mean NDVI time series
(which shows the dip when the clearing appears), per-pixel median composites for
the baseline and recent periods, the change map, the loss/stable/gain
classification, and the hectares. The headline question — does the detector
recover the planted clearing? — is answered by ``planted_loss_recovered``, the
fraction of the planted block flagged as loss.

Everything is deterministic given the seed (the rng only adds noise and scatters
clouds), so the numbers are stable enough to pin in a unit test and quote in the
README. It is honest about what it is: a seeded synthetic stack, not a real S2
run. The point is to exercise the change detection the real pipeline depends on,
with a one-command reproducible artifact.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from eets.change import change_map, change_stats, classify_change
from eets.timeseries import index_timeseries, mask_invalid, temporal_composite

# Scene geometry. 80x80 pixels at 10 m/px is an 800 m square — large enough to
# hold a sizeable clearing while the whole demo runs in well under a second.
ROWS = 80
COLS = 80
PIXEL_SIZE_M = 10.0

# Time axis: 6 baseline scenes then 6 recent scenes (e.g. 2018-2019 vs 2023-2024).
N_BASELINE = 6
N_RECENT = 6

# Background (intact forest) NDVI and the cleared-block NDVI after planting.
FOREST_NDVI = 0.85
CLEARED_NDVI = 0.25

# The planted clearing: a rectangular block that appears in the recent period.
CLEAR_R0, CLEAR_R1 = 20, 50  # 30 rows
CLEAR_C0, CLEAR_C1 = 30, 60  # 30 cols  -> 900 px = 9 ha at 10 m

# Change thresholds on NDVI delta (after - before).
LOSS_THRESH = -0.20
GAIN_THRESH = 0.20

# SCL classes treated as invalid (synthetic clouds use class 9 = cloud high-prob).
INVALID_SCL = (3, 8, 9, 10)
CLOUD_CLASS = 9
N_CLOUD_PIXELS = 15  # scattered cloudy pixels per scene


def _synthesize_stack(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the synthetic NDVI stack, the SCL stack, and the planted-block mask.

    Returns ``(ndvi_stack, scl_stack, planted_mask)`` where ``ndvi_stack`` is
    ``(time, row, col)`` clean NDVI, ``scl_stack`` carries the synthetic clouds,
    and ``planted_mask`` is the boolean ground-truth clearing.
    """
    rng = np.random.default_rng(seed)
    n_time = N_BASELINE + N_RECENT
    ndvi_stack = np.full((n_time, ROWS, COLS), FOREST_NDVI, dtype=np.float64)

    planted_mask = np.zeros((ROWS, COLS), dtype=bool)
    planted_mask[CLEAR_R0:CLEAR_R1, CLEAR_C0:CLEAR_C1] = True

    # Recent scenes: drop NDVI inside the clearing to the cleared level.
    for t in range(N_BASELINE, n_time):
        ndvi_stack[t][planted_mask] = CLEARED_NDVI

    # Mild per-scene noise so composites are not trivially constant.
    ndvi_stack += rng.normal(0.0, 0.02, size=ndvi_stack.shape)
    ndvi_stack = np.clip(ndvi_stack, -1.0, 1.0)

    # SCL: start all-vegetation (class 4), scatter a few cloud pixels per scene.
    scl_stack = np.full((n_time, ROWS, COLS), 4, dtype=np.int16)
    for t in range(n_time):
        rr = rng.integers(0, ROWS, size=N_CLOUD_PIXELS)
        cc = rng.integers(0, COLS, size=N_CLOUD_PIXELS)
        scl_stack[t, rr, cc] = CLOUD_CLASS

    return ndvi_stack, scl_stack, planted_mask


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the dependency-free time-series + change demo and write artifacts.

    Synthesises the multi-year NDVI stack with a planted clearing, masks the
    synthetic clouds with the SCL, computes the spatial-mean NDVI time series,
    builds baseline and recent median composites, maps and classifies the change,
    and totals the hectares. Writes ``index_timeseries.csv`` (one row per scene)
    and ``change_stats.json`` (the returned dict) into ``out_dir``.

    Parameters
    ----------
    seed:
        Seed for the per-scene noise and cloud scatter. The scene layout is fixed.
    out_dir:
        Directory for the artifacts. Created if missing.

    Returns
    -------
    dict
        Keys: ``n_timesteps``, ``baseline_mean_ndvi``, ``recent_mean_ndvi``,
        ``loss_hectares``, ``gain_hectares``, ``planted_loss_recovered``
        (fraction of the planted block flagged as loss), plus the pixel size and
        thresholds used.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ndvi_stack, scl_stack, planted_mask = _synthesize_stack(seed)

    # Cloud-mask every scene: invalid SCL pixels -> NaN before any reduction.
    masked = np.stack(
        [
            mask_invalid(ndvi_stack[t], scl_stack[t], INVALID_SCL)
            for t in range(ndvi_stack.shape[0])
        ],
        axis=0,
    )

    # 1) Spatial-mean NDVI time series (NaN-aware) — shows the dip at clearing.
    series = index_timeseries(masked, axis=0)

    # 2) Baseline vs recent median composites (cloud-robust).
    before = temporal_composite(masked[:N_BASELINE], agg="median", axis=0)
    after = temporal_composite(masked[N_BASELINE:], agg="median", axis=0)

    # 3) Change map -> classify -> hectares.
    delta = change_map(before, after)
    classified = classify_change(delta, LOSS_THRESH, GAIN_THRESH)
    stats = change_stats(classified, PIXEL_SIZE_M)

    # 4) How much of the planted clearing did we recover as loss?
    planted_total = int(planted_mask.sum())
    planted_loss = int(np.count_nonzero((classified == -1) & planted_mask))
    planted_loss_recovered = planted_loss / planted_total if planted_total else 0.0

    baseline_mean_ndvi = float(np.nanmean(series[:N_BASELINE]))
    recent_mean_ndvi = float(np.nanmean(series[N_BASELINE:]))

    summary: dict[str, Any] = {
        "seed": int(seed),
        "raster": [ROWS, COLS],
        "pixel_size_m": float(PIXEL_SIZE_M),
        "n_timesteps": int(masked.shape[0]),
        "n_baseline": int(N_BASELINE),
        "n_recent": int(N_RECENT),
        "loss_thresh": float(LOSS_THRESH),
        "gain_thresh": float(GAIN_THRESH),
        "baseline_mean_ndvi": baseline_mean_ndvi,
        "recent_mean_ndvi": recent_mean_ndvi,
        "loss_pixels": stats["loss_pixels"],
        "gain_pixels": stats["gain_pixels"],
        "loss_hectares": stats["loss_hectares"],
        "gain_hectares": stats["gain_hectares"],
        "planted_block_hectares": planted_total * PIXEL_SIZE_M**2 / 10_000.0,
        "planted_loss_recovered": planted_loss_recovered,
    }

    # index_timeseries.csv: scene index, period label, mean NDVI.
    with open(out / "index_timeseries.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestep", "period", "mean_ndvi"])
        for t, v in enumerate(series):
            period = "baseline" if t < N_BASELINE else "recent"
            writer.writerow([t, period, f"{v:.6f}"])

    with open(out / "change_stats.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
