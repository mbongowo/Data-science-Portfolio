"""floodmap: SAR flood mapping & before/after water-change in hectares.

This package turns a before/after pair of Sentinel-1 SAR backscatter scenes into
*flood numbers*: an automatic Otsu water threshold per scene, a boolean water
mask, the before/after flood-extent change (newly flooded, permanent water,
receded), and the hectares of each. That quantification — the threshold, the
masks, the flood extent, the hectares — is the runnable, tested contribution.

The numeric core (:mod:`floodmap.threshold`, :mod:`floodmap.water`,
:mod:`floodmap.change`) and the demo (:mod:`floodmap.demo`) depend only on numpy
and the standard library, so they import and test anywhere. The default real-data
path (:mod:`floodmap.stac`, Earth Search Sentinel-1 GRD / Sentinel-2 L2A, no
auth) imports its heavy dependencies lazily, inside functions, and is not
imported here or by the test suite.
"""

from __future__ import annotations

from floodmap.change import flood_extent, flood_stats
from floodmap.demo import run_demo
from floodmap.threshold import histogram_modes, otsu_threshold
from floodmap.water import mndwi, to_db, water_mask

__all__ = [
    "otsu_threshold",
    "histogram_modes",
    "water_mask",
    "mndwi",
    "to_db",
    "flood_extent",
    "flood_stats",
    "run_demo",
    "__version__",
]

__version__ = "0.1.0"
