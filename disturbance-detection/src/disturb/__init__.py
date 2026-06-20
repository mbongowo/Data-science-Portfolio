"""disturb: time-series change & disturbance detection for satellite NDVI cubes.

Pure-numpy core (harmonic decomposition + CUSUM breakpoint detection) imports
without any geospatial dependencies. Heavy / optional dependencies (odc-stac,
xarray, rioxarray, statsmodels, ruptures) are imported lazily inside the
functions that need them so that ``import disturb`` always succeeds.
"""

from __future__ import annotations

from .decompose import HarmonicFit, harmonic_decompose
from .detect import Breakpoint, detect_breakpoint

__all__ = [
    "HarmonicFit",
    "harmonic_decompose",
    "Breakpoint",
    "detect_breakpoint",
]

__version__ = "0.1.0"
