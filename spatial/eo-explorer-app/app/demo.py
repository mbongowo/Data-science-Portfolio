"""Offline, deterministic demo for the EO Explorer app.

This runs the *pure* core of the app end to end with **no network and no STAC**:
it validates a small synthetic AOI, synthesises band arrays for one scene with a
seeded generator, computes NDVI / NDWI / NDMI with the same index functions the
app uses, and reports real summary numbers. If matplotlib and Pillow happen to be
installed it also colourises NDVI and writes a PNG; if they are not, the demo
still produces every number and simply skips the image.

Run it with::

    python -m app.demo

It writes ``summary.json`` (and, when possible, ``ndvi.png``) into ``outputs/``
and prints the summary dict. Everything is seeded, so the metrics are
reproducible and are asserted in ``tests/test_demo.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app import render, stac

#: The synthetic AOI used by the demo: a small box near 50N, well under the area
#: cap so ``validate_aoi`` accepts it.
DEMO_BBOX: tuple[float, float, float, float] = (10.0, 50.0, 10.1, 50.1)

#: The (fixed) date the demo pretends to query, used only for the cache key.
DEMO_DATE = "2024-06-15"

#: Synthetic scene size in pixels (small so the demo runs in well under a second).
DEMO_SHAPE = (64, 64)


def _synthesize_bands(seed: int, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Make deterministic red/nir/green/swir reflectance arrays in ``[0, 1]``.

    The values are drawn from a seeded generator so the whole demo is
    reproducible. A mild vegetation-like structure (high NIR, low red over part
    of the scene) keeps the resulting indices in a believable range, but the
    exact numbers only need to be *stable*, not physical.
    """
    rng = np.random.default_rng(seed)
    red = rng.uniform(0.02, 0.20, size=shape)
    nir = rng.uniform(0.30, 0.60, size=shape)
    green = rng.uniform(0.05, 0.25, size=shape)
    swir = rng.uniform(0.10, 0.40, size=shape)
    return {"red": red, "nir": nir, "green": green, "swir16": swir}


def run_demo(seed: int = 0, out_dir: str = "outputs") -> dict[str, Any]:
    """Run the offline demo and return a dict of real metrics.

    Parameters
    ----------
    seed : int, optional
        Seed for the synthetic-band generator. The default (0) produces the
        numbers committed in the README and asserted in the tests.
    out_dir : str, optional
        Directory for ``summary.json`` and (when matplotlib + Pillow are
        present) ``ndvi.png``. Created if it does not exist.

    Returns
    -------
    dict
        Keys: ``bbox``, ``area_km2``, ``aoi_ok``, ``ndvi_mean``,
        ``ndvi_valid_fraction``, ``cache_key``, ``png_written`` (bool).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Validate the AOI -- the same gate the UI uses.
    validation = stac.validate_aoi(DEMO_BBOX)

    # 2. Synthesize one scene's worth of bands.
    bands = _synthesize_bands(seed, DEMO_SHAPE)

    # 3. Compute the three indices via the render fallback functions
    #    (numerically identical to eo-monitor's).
    ndvi = render._ndvi(bands["nir"], bands["red"])
    ndwi = render._ndwi(bands["green"], bands["nir"])
    ndmi = render._ndmi(bands["nir"], bands["swir16"])

    ndvi_stats = render.index_stats(ndvi)
    ndwi_stats = render.index_stats(ndwi)
    ndmi_stats = render.index_stats(ndmi)

    # A normalized array + a robust stretch, to show the preview helpers.
    spec = render.INDEX_REGISTRY["NDVI"]
    normalized = render.normalize(ndvi, vmin=spec.vmin, vmax=spec.vmax)
    stretch = render.percentile_stretch(ndvi)

    # 4. Build the deterministic cache key for this request.
    key = stac.cache_key(DEMO_BBOX, DEMO_DATE, "NDVI")

    # 5. Optional: colourise NDVI to a PNG if matplotlib + Pillow are available.
    png_written = False
    png_path = out_path / "ndvi.png"
    try:
        from PIL import Image  # noqa: F401

        rgba = render.colorize(ndvi, vmin=spec.vmin, vmax=spec.vmax, colormap=spec.colormap)
        Image.fromarray(rgba, mode="RGBA").save(png_path, format="PNG")
        png_written = True
    except Exception:  # noqa: BLE001 - matplotlib/PIL optional; demo works without
        png_written = False

    result = {
        "bbox": list(DEMO_BBOX),
        "area_km2": round(validation.area_km2, 6),
        "aoi_ok": validation.ok,
        "ndvi_mean": round(ndvi_stats["mean"], 6),
        "ndvi_valid_fraction": ndvi_stats["valid_fraction"],
        "cache_key": key,
        "png_written": png_written,
    }

    summary = {
        **result,
        "seed": seed,
        "shape": list(DEMO_SHAPE),
        "date": DEMO_DATE,
        "center": stac.bbox_center(DEMO_BBOX),
        "aspect_ratio": round(stac.bbox_aspect_ratio(DEMO_BBOX), 6),
        "suggest_zoom": stac.suggest_zoom(DEMO_BBOX),
        "ndvi_stretch_p2_p98": [round(stretch[0], 6), round(stretch[1], 6)],
        "normalized_mean": round(float(np.nanmean(normalized)), 6),
        "stats": {"NDVI": ndvi_stats, "NDWI": ndwi_stats, "NDMI": ndmi_stats},
        "aoi_message": validation.message,
        "png_path": str(png_path) if png_written else None,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """``python -m app.demo`` entry point: run the demo and print the metrics."""
    result = run_demo(seed=0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
