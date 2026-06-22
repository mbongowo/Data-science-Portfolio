"""A self-contained, dependency-free ESDA demo on a seeded synthetic grid.

This module exists so the package can be run end to end without the geospatial
stack, without a network call, and without any downloaded data. It builds a
rook-contiguity grid by hand, plants a High-High cluster and a Low-Low cluster
into an otherwise noisy field, and drives the *real* pure-numpy ESDA core:
global Moran's I, Geary's C, the LISA quadrant labels, and the standardised
Getis-Ord Gi*. Everything is deterministic given the seed, so the numbers are
stable enough to pin in a unit test and to quote in the README.

It is honest about what it is: a small seeded synthetic field on a regular grid,
not a real dataset. The point is to show the statistics responding to structure
that was deliberately planted, and to give a one-command reproducible artifact.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hotspots.esda import (
    expected_morans_i,
    gearys_c_dense,
    getis_ord_g_star_dense,
    lisa_quadrants,
    morans_i_dense,
    rook_weights,
)

# Grid geometry for the demo. A 12 x 12 rook grid is large enough for the
# planted clusters and the asymptotic Gi* z-scores to be meaningful while still
# running in well under a second.
GRID_ROWS = 12
GRID_COLS = 12

# Gi* hot/cold thresholds: |z| > 1.96 is the usual two-sided 5% asymptotic cut.
GI_THRESHOLD = 1.96


def _synthesize_field(seed: int) -> np.ndarray:
    """Build the synthetic field: a high block, a low block, and Gaussian noise.

    The field is a 12 x 12 grid flattened row-major (matching
    :func:`hotspots.esda.rook_weights`). A High-High block is planted in the
    top-left, a Low-Low block in the bottom-right, and zero-mean noise is added
    everywhere so the clusters are real but not trivial.
    """
    rng = np.random.default_rng(seed)
    field = rng.normal(loc=0.0, scale=1.0, size=(GRID_ROWS, GRID_COLS))

    # Planted High-High cluster (top-left 4x4 block) and Low-Low cluster
    # (bottom-right 4x4 block). The offsets dominate the unit-scale noise.
    field[0:4, 0:4] += 6.0
    field[GRID_ROWS - 4 : GRID_ROWS, GRID_COLS - 4 : GRID_COLS] -= 6.0
    return field.ravel()


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the dependency-free ESDA demo and write artifacts.

    Builds a rook-contiguity weights matrix for a 12 x 12 grid, synthesizes a
    field with a planted High-High and Low-Low cluster plus noise, and computes
    the global and local statistics from the pure-numpy core. Writes
    ``lisa_labels.csv`` (per-cell row, col, value, lisa label) and
    ``summary.json`` (the returned dict) into ``out_dir``.

    Parameters
    ----------
    seed:
        Seed for the synthetic field. The grid and weights are fixed; only the
        noise depends on the seed.
    out_dir:
        Directory for the artifacts. Created if missing.

    Returns
    -------
    dict
        Keys: ``n``, ``morans_i``, ``expected_i``, ``gearys_c``,
        ``lisa_counts`` (a dict with HH/LL/LH/HL/ns), ``gi_hot``, ``gi_cold``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    values = _synthesize_field(seed)
    n = values.size
    w = rook_weights(GRID_ROWS, GRID_COLS)

    morans_i = morans_i_dense(values, w)
    expected_i = expected_morans_i(n)
    gearys_c = gearys_c_dense(values, w)

    labels = lisa_quadrants(values, w)
    # The pure-numpy quadrant labeller emits "00" for the undefined quadrant; the
    # demo reports counts under the conventional HH/LL/LH/HL plus an "ns" bucket
    # that collects everything without a strict quadrant.
    counts = Counter(labels.tolist())
    lisa_counts = {
        "HH": int(counts.get("HH", 0)),
        "LL": int(counts.get("LL", 0)),
        "LH": int(counts.get("LH", 0)),
        "HL": int(counts.get("HL", 0)),
        "ns": int(counts.get("00", 0)),
    }

    gi_z = getis_ord_g_star_dense(values, w)
    gi_hot = int(np.sum(gi_z > GI_THRESHOLD))
    gi_cold = int(np.sum(gi_z < -GI_THRESHOLD))

    summary: dict[str, Any] = {
        "n": int(n),
        "grid": [GRID_ROWS, GRID_COLS],
        "seed": int(seed),
        "morans_i": float(morans_i),
        "expected_i": float(expected_i),
        "gearys_c": float(gearys_c),
        "lisa_counts": lisa_counts,
        "gi_hot": gi_hot,
        "gi_cold": gi_cold,
    }

    # Per-cell LISA labels, for a map or a sanity check.
    with open(out / "lisa_labels.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "row", "col", "value", "lisa_label"])
        for i in range(n):
            r, c = divmod(i, GRID_COLS)
            writer.writerow([i, r, c, f"{values[i]:.6f}", labels[i]])

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
