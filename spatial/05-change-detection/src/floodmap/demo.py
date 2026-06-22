"""A self-contained, dependency-free demo of the SAR flood-mapping core.

This module runs the whole project end to end without the STAC stack, without a
network call, and without any downloaded imagery. It deterministically
synthesises a small before/after pair of Sentinel-1-like backscatter scenes over
a land area: dry land has high backscatter, a permanent **river** runs through
the scene as low backscatter in *both* dates, and a rectangular **planted flood**
block is dry land in ``pre`` but becomes water (low backscatter) in ``post``.
Multiplicative speckle noise — the characteristic SAR grain — is added to both.

It then drives the *real* pure-numpy core: :func:`floodmap.water.to_db` to dB,
:func:`floodmap.threshold.otsu_threshold` to find the water cut-off on each date,
:func:`floodmap.water.water_mask` (``polarity="below"``) for the water masks,
:func:`floodmap.change.flood_extent` for the before/after split, and
:func:`floodmap.change.flood_stats` for the hectares. The headline question — does
the mapper recover the planted flood? — is answered by
``planted_flood_recovered``, the fraction of the planted block flagged as flooded.

Everything is deterministic given the seed (the rng only adds speckle), so the
numbers are stable enough to pin in a unit test and quote in the README. It is
honest about what it is: a seeded synthetic pair, not a real Sentinel-1 run. The
point is to exercise the flood mapping the real pipeline depends on, with a
one-command reproducible artifact. Runs in well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from floodmap.change import flood_extent, flood_stats
from floodmap.threshold import otsu_threshold
from floodmap.water import to_db, water_mask

# Scene geometry. 100x100 pixels at 10 m/px is a 1 km square — large enough to
# hold a sizeable flood while the whole demo runs in well under a second.
ROWS = 100
COLS = 100
PIXEL_SIZE_M = 10.0

# Linear backscatter levels (sigma-nought). Dry land scatters strongly; smooth
# water scatters radar away from the sensor, so its return is ~an order of
# magnitude lower. These translate to ~ -7 dB land and ~ -19 dB water.
LAND_BACKSCATTER = 0.20
WATER_BACKSCATTER = 0.012

# Permanent river: a vertical channel low in both dates (columns 46-50).
RIVER_C0, RIVER_C1 = 46, 50

# The planted flood: a rectangular block, dry land in `pre`, water in `post`.
FLOOD_R0, FLOOD_R1 = 30, 70  # 40 rows
FLOOD_C0, FLOOD_C1 = 60, 90  # 30 cols -> 1200 px = 12 ha at 10 m

# SAR speckle is multiplicative; this is its relative standard deviation.
SPECKLE_REL_STD = 0.25


def _synthesize_pair(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the synthetic pre/post linear-backscatter pair and the flood mask.

    Returns ``(pre_linear, post_linear, planted_flood_mask)`` where the two
    arrays are ``(row, col)`` linear backscatter and ``planted_flood_mask`` is
    the boolean ground-truth flood block.
    """
    rng = np.random.default_rng(seed)

    pre = np.full((ROWS, COLS), LAND_BACKSCATTER, dtype=np.float64)
    post = np.full((ROWS, COLS), LAND_BACKSCATTER, dtype=np.float64)

    # Permanent river: low backscatter in both dates.
    pre[:, RIVER_C0:RIVER_C1] = WATER_BACKSCATTER
    post[:, RIVER_C0:RIVER_C1] = WATER_BACKSCATTER

    # Planted flood: dry in pre, water in post.
    planted_flood = np.zeros((ROWS, COLS), dtype=bool)
    planted_flood[FLOOD_R0:FLOOD_R1, FLOOD_C0:FLOOD_C1] = True
    post[planted_flood] = WATER_BACKSCATTER

    # Multiplicative speckle (clipped positive so to_db stays finite).
    pre *= 1.0 + rng.normal(0.0, SPECKLE_REL_STD, size=pre.shape)
    post *= 1.0 + rng.normal(0.0, SPECKLE_REL_STD, size=post.shape)
    pre = np.clip(pre, 1e-4, None)
    post = np.clip(post, 1e-4, None)

    return pre, post, planted_flood


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the dependency-free synthetic flood-mapping demo and write artifacts.

    Synthesises the before/after SAR backscatter pair (permanent river + planted
    flood + speckle), converts to dB, Otsu-thresholds each date, builds the water
    masks, computes the before/after flood extent, and totals the hectares.
    Writes ``flood_stats.json`` (the returned dict) and ``flood_mask.npy`` (the
    post-flood ``flooded`` boolean mask) into ``out_dir``.

    Parameters
    ----------
    seed:
        Seed for the speckle noise. The scene layout is fixed.
    out_dir:
        Directory for the artifacts. Created if missing.

    Returns
    -------
    dict
        Keys include ``pre_water_threshold_db``, ``post_water_threshold_db``,
        ``flooded_hectares``, ``permanent_water_hectares``, and
        ``planted_flood_recovered`` (fraction of the planted block flagged as
        flooded), plus the pixel size, raster shape, and seed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pre_linear, post_linear, planted_flood = _synthesize_pair(seed)

    # 1) Linear backscatter -> decibels (Otsu is done on dB).
    pre_db = to_db(pre_linear)
    post_db = to_db(post_linear)

    # 2) Otsu water threshold per date (automatic, no hand-tuned cut-off).
    pre_thresh = otsu_threshold(pre_db)
    post_thresh = otsu_threshold(post_db)

    # 3) Water masks: SAR water is dark, so water is *below* the threshold.
    pre_water = water_mask(pre_db, pre_thresh, polarity="below")
    post_water = water_mask(post_db, post_thresh, polarity="below")

    # 4) Before/after flood extent -> hectares.
    masks = flood_extent(pre_water, post_water)
    stats = flood_stats(masks, PIXEL_SIZE_M)

    # 5) How much of the planted flood did we recover?
    planted_total = int(planted_flood.sum())
    planted_hit = int(np.count_nonzero(masks["flooded"] & planted_flood))
    planted_flood_recovered = planted_hit / planted_total if planted_total else 0.0

    summary: dict[str, Any] = {
        "seed": int(seed),
        "raster": [ROWS, COLS],
        "pixel_size_m": float(PIXEL_SIZE_M),
        "pre_water_threshold_db": float(pre_thresh),
        "post_water_threshold_db": float(post_thresh),
        "flooded_pixels": stats["flooded_pixels"],
        "flooded_hectares": stats["flooded_hectares"],
        "permanent_water_hectares": stats["permanent_water_hectares"],
        "receded_hectares": stats["receded_hectares"],
        "flooded_fraction_of_scene": stats["flooded_fraction_of_scene"],
        "planted_flood_hectares": planted_total * PIXEL_SIZE_M**2 / 10_000.0,
        "planted_flood_recovered": planted_flood_recovered,
    }

    with open(out / "flood_stats.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.save(out / "flood_mask.npy", masks["flooded"])

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
