"""spatial-hotspots: exploratory spatial data analysis (ESDA).

This package builds spatial weights matrices, tests for global spatial
autocorrelation, maps local clusters and spatial outliers (LISA / Getis-Ord
Gi*), and optionally fits a Geographically Weighted Regression (GWR).

The package is split so that the interpretation-critical numeric core (a
pure-numpy reference implementation of Moran's I) has no third-party
dependency and is always importable and testable.
"""

from __future__ import annotations

from hotspots.esda import (
    expected_morans_i,
    gearys_c_dense,
    getis_ord_g_star_dense,
    lisa_quadrants,
    local_moran_dense,
    morans_i_dense,
)

__all__ = [
    "morans_i_dense",
    "expected_morans_i",
    "gearys_c_dense",
    "local_moran_dense",
    "lisa_quadrants",
    "getis_ord_g_star_dense",
    "__version__",
]

__version__ = "0.1.0"
