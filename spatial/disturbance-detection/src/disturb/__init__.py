"""disturb: time-series change & disturbance detection for satellite NDVI cubes.

Pure-numpy core (harmonic decomposition + CUSUM breakpoint detection) imports
without any geospatial dependencies. Heavy / optional dependencies (odc-stac,
xarray, rioxarray, statsmodels, ruptures) are imported lazily inside the
functions that need them so that ``import disturb`` always succeeds.
"""

from __future__ import annotations

from .decompose import HarmonicFit, harmonic_decompose
from .demo import run_demo
from .detect import (
    Breakpoint,
    detect_breakpoint,
    detect_breakpoints_binseg,
    recovery_time,
)
from .trend import MannKendallResult, mann_kendall, theil_sen_slope

__all__ = [
    "HarmonicFit",
    "harmonic_decompose",
    "Breakpoint",
    "detect_breakpoint",
    "detect_breakpoints_binseg",
    "recovery_time",
    "theil_sen_slope",
    "mann_kendall",
    "MannKendallResult",
    "run_demo",
]

__version__ = "0.1.0"
