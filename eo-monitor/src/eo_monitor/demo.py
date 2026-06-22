"""Self-contained, offline demo driving the real index + anomaly core.

``run_demo`` synthesises a tiny Sentinel-2-like reflectance cube with numpy's
``default_rng`` (so it is fully deterministic for a given seed), plants a
vegetation-loss patch in the target window, then runs the *real*
:mod:`eo_monitor.indices` and :mod:`eo_monitor.anomaly` code to recover it. No
network, no GDAL, no STAC — pure numpy, well under a second.

The planted region is a rectangular block where the target NIR is depressed
(canopy loss), which drives NDVI down and produces a strong negative z-score
there. The demo reports how well the |z| > 2 anomaly mask recovers that block.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eo_monitor.anomaly import anomaly_cube, anomaly_fraction, classify_anomaly
from eo_monitor.indices import ndmi, ndvi, ndwi

# Demo grid + window sizes. Small enough to be instant, large enough that the
# planted block and the anomaly statistics are meaningful.
GRID = (24, 24)  # (y, x)
BASELINE_TIMES = 8
TARGET_TIMES = 4
# Planted vegetation-loss block (row/col half-open slices into the grid).
PLANTED = (slice(6, 14), slice(8, 18))
Z_THRESHOLD = 2.0


def _synthesize(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(baseline_bands, target_bands, planted_mask)``.

    Each ``*_bands`` is a ``(nir, red, swir, green)`` tuple of ``(time, y, x)``
    reflectance cubes. ``planted_mask`` is the ``(y, x)`` boolean block where the
    target vegetation was knocked down.
    """
    rng = np.random.default_rng(seed)
    ny, nx = GRID

    # Healthy-vegetation reflectances (fractions): high NIR, low red, moderate
    # green/SWIR, with small per-scene noise so the baseline has real variance.
    def stack(times: int, nir0: float, red0: float, green0: float, swir0: float):
        nir = nir0 + rng.normal(0.0, 0.02, size=(times, ny, nx))
        red = red0 + rng.normal(0.0, 0.01, size=(times, ny, nx))
        green = green0 + rng.normal(0.0, 0.01, size=(times, ny, nx))
        swir = swir0 + rng.normal(0.0, 0.02, size=(times, ny, nx))
        return nir, red, swir, green

    base = stack(BASELINE_TIMES, 0.45, 0.06, 0.10, 0.20)
    tgt = stack(TARGET_TIMES, 0.45, 0.06, 0.10, 0.20)

    # Plant the anomaly: NIR collapses and red rises in the block for the target
    # window only -> NDVI drops sharply there.
    nir_t, red_t, swir_t, green_t = tgt
    ry, rx = PLANTED
    nir_t[:, ry, rx] -= 0.28
    red_t[:, ry, rx] += 0.06
    swir_t[:, ry, rx] += 0.06  # drier canopy too

    planted_mask = np.zeros(GRID, dtype=bool)
    planted_mask[ry, rx] = True

    return base, (nir_t, red_t, swir_t, green_t), planted_mask


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the offline demo and write artifacts; return a metrics dict.

    Parameters
    ----------
    seed
        Seed for the deterministic synthetic cube.
    out_dir
        Directory for ``ndvi_demo.npy`` (target NDVI map), ``ndvi_zscore.npy``
        (the z-score field) and ``summary.json`` (the returned metrics).

    Returns
    -------
    dict with keys: ``seed``, ``grid_shape``, ``mean_ndvi``,
    ``anomaly_pixel_count``, ``anomaly_fraction``, ``max_abs_z``,
    ``planted_pixel_count``, ``planted_region_recovered`` (recall of the
    planted block by the |z| > 2 loss mask).
    """
    base, tgt, planted_mask = _synthesize(seed)
    nir_b, red_b, swir_b, green_b = base
    nir_t, red_t, swir_t, green_t = tgt

    # Real core: per-scene indices, then median-composite the target window.
    ndvi_base = ndvi(nir_b, red_b)
    ndvi_target = np.nanmedian(ndvi(nir_t, red_t), axis=0)
    # Companion indices, computed so the demo exercises the full set.
    ndwi_target = np.nanmedian(ndwi(green_t, nir_t), axis=0)
    ndmi_target = np.nanmedian(ndmi(nir_t, swir_t), axis=0)

    # z-score anomaly of the target NDVI composite vs the baseline stack.
    z = anomaly_cube(ndvi_target, ndvi_base)

    # Loss classification: -1 where z < -2. Recovery = recall of planted block.
    klass = classify_anomaly(z, threshold=Z_THRESHOLD)
    loss_mask = klass == -1
    planted_count = int(planted_mask.sum())
    recovered = int((loss_mask & planted_mask).sum())
    recall = recovered / planted_count if planted_count else 0.0

    finite_z = z[np.isfinite(z)]
    metrics = {
        "seed": int(seed),
        "grid_shape": list(GRID),
        "mean_ndvi": round(float(np.nanmean(ndvi_target)), 6),
        "mean_ndwi": round(float(np.nanmean(ndwi_target)), 6),
        "mean_ndmi": round(float(np.nanmean(ndmi_target)), 6),
        "anomaly_pixel_count": int(np.sum(np.abs(z) > Z_THRESHOLD)),
        "anomaly_fraction": round(anomaly_fraction(z, Z_THRESHOLD), 6),
        "max_abs_z": round(float(np.max(np.abs(finite_z))), 6),
        "planted_pixel_count": planted_count,
        "planted_region_recovered": round(recall, 6),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ndvi_demo.npy", ndvi_target)
    np.save(out / "ndvi_zscore.npy", z)
    (out / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":  # pragma: no cover
    import pprint

    pprint.pprint(run_demo(0))
