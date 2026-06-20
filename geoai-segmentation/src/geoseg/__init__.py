"""geoseg: reproducible deep-learning semantic segmentation for Earth-observation imagery.

The package is designed so that *importing it never pulls in heavy optional
dependencies* (torch, lightning, rasterio, ...). Pure-python / numpy utilities
(metrics, deterministic split logic) are always importable, which keeps the test
suite runnable on a bare CI machine.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
