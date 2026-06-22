"""A self-contained, dependency-free demo of the mask analytics core.

This module exists so the package can be run end to end without the geospatial
or deep-learning stack, without a GPU, without a network call, and without any
downloaded imagery. It deterministically synthesises a small label raster that
stands in for a Segment Anything output over a Douala neighbourhood: a regular
grid of rectangular "building" footprints of varied sizes, plus one large
"field" polygon. It then drives the *real* pure-numpy core — connected-component
labelling, region properties, area filtering to the building-size band, and the
pixel-to-metre conversions — to count the buildings, measure their footprints,
and total the field area.

Everything is deterministic given the seed (the rng only jitters footprint
sizes), so the headline numbers are stable enough to pin in a unit test and to
quote in the README. It is honest about what it is: a seeded synthetic raster,
not a real SAM run on Douala imagery. The point is to exercise the quantification
that the real pipeline depends on, with a one-command reproducible artifact.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from samgeo_post.analytics import (
    area_hectares,
    count_objects,
    filter_by_area,
    label_components,
    mask_iou,
    pixels_to_area,
    region_props,
)

# Raster geometry for the demo. A 120 x 120 grid is large enough to hold a 5x5
# block of well-separated buildings plus a sizeable field while running in well
# under a second.
RASTER_ROWS = 120
RASTER_COLS = 120

# Ground sample distance. 0.5 m/px is a realistic high-resolution basemap tile
# resolution, so footprints land in plausible square-metre ranges.
PIXEL_SIZE_M = 0.5

# Building-size band in pixels. Buildings here span ~8x8 to ~14x14 px (64-196
# px); the band keeps that range, dropping speckle below and the field above.
BUILDING_MIN_PX = 30
BUILDING_MAX_PX = 400


def _synthesize_labels(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the synthetic scene and return ``(binary_mask, true_labels)``.

    Lays out a 5 x 5 grid of rectangular building footprints with rng-jittered
    sizes in the top portion of the raster, plus one large rectangular field in
    the bottom portion. ``true_labels`` is the ground-truth labelling used only
    for the IoU sanity check; ``binary_mask`` is what the core re-derives.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((RASTER_ROWS, RASTER_COLS), dtype=np.uint8)

    # 5x5 block of buildings on a 16-pixel grid pitch, well separated so that
    # 4-connectivity keeps them distinct. Footprint sides jitter from 8 to 13 px,
    # leaving at least a 3-pixel gap between neighbours.
    n_side = 5
    pitch = 16
    origin_r, origin_c = 6, 6
    for i in range(n_side):
        for j in range(n_side):
            h = int(rng.integers(8, 14))
            w = int(rng.integers(8, 14))
            r0 = origin_r + i * pitch
            c0 = origin_c + j * pitch
            mask[r0 : r0 + h, c0 : c0 + w] = 1

    # One large field polygon in the lower strip, separated from the buildings.
    mask[92:115, 10:110] = 1

    true_labels = label_components(mask, connectivity=4)
    return mask, true_labels


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the dependency-free analytics demo and write artifacts.

    Synthesises the building/field raster, labels it with the pure-numpy core,
    computes region properties, splits objects into the building-size band and
    the oversized "field" remainder, and converts pixel areas to square metres
    and hectares using :data:`PIXEL_SIZE_M`. Writes ``region_props.csv`` (one
    row per labelled object with its area in px and m**2) and ``summary.json``
    (the returned dict) into ``out_dir``.

    Parameters
    ----------
    seed:
        Seed for the footprint-size jitter. The layout is otherwise fixed.
    out_dir:
        Directory for the artifacts. Created if missing.

    Returns
    -------
    dict
        Keys: ``n_objects``, ``n_buildings``, ``mean_building_footprint_m2``,
        ``total_building_area_m2``, ``field_area_hectares``, ``pixel_size_m``,
        ``reconstruction_iou`` (1.0 when the labelling round-trips the mask).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask, labeled = _synthesize_labels(seed)
    props = region_props(labeled)
    n_objects = count_objects(labeled)

    buildings = filter_by_area(props, min_px=BUILDING_MIN_PX, max_px=BUILDING_MAX_PX)
    fields = filter_by_area(props, min_px=BUILDING_MAX_PX + 1)

    building_areas_m2 = [pixels_to_area(p["area_px"], PIXEL_SIZE_M) for p in buildings]
    total_building_area_m2 = float(sum(building_areas_m2))
    n_buildings = len(buildings)
    mean_building_footprint_m2 = (
        total_building_area_m2 / n_buildings if n_buildings else 0.0
    )

    field_area_px = sum(p["area_px"] for p in fields)
    field_area_hectares = area_hectares(pixels_to_area(field_area_px, PIXEL_SIZE_M))

    # Sanity check: re-paint every labelled object back onto an empty raster and
    # confirm it reconstructs the input mask exactly (IoU == 1.0).
    reconstructed = (labeled > 0).astype(np.uint8)
    reconstruction_iou = mask_iou(reconstructed, mask)

    summary: dict[str, Any] = {
        "seed": int(seed),
        "raster": [RASTER_ROWS, RASTER_COLS],
        "pixel_size_m": float(PIXEL_SIZE_M),
        "n_objects": int(n_objects),
        "n_buildings": int(n_buildings),
        "mean_building_footprint_m2": float(mean_building_footprint_m2),
        "total_building_area_m2": float(total_building_area_m2),
        "field_area_hectares": float(field_area_hectares),
        "reconstruction_iou": float(reconstruction_iou),
    }

    with open(out / "region_props.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["label", "area_px", "area_m2", "rmin", "cmin", "rmax", "cmax", "class"]
        )
        for p in props:
            a = p["area_px"]
            cls = (
                "building"
                if BUILDING_MIN_PX <= a <= BUILDING_MAX_PX
                else ("field" if a > BUILDING_MAX_PX else "noise")
            )
            rmin, cmin, rmax, cmax = p["bbox"]
            writer.writerow(
                [
                    p["label"],
                    a,
                    f"{pixels_to_area(a, PIXEL_SIZE_M):.4f}",
                    rmin,
                    cmin,
                    rmax,
                    cmax,
                    cls,
                ]
            )

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
