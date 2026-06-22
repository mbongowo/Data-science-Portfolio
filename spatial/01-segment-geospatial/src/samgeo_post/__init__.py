"""samgeo_post: geospatial post-processing and quantification of SAM masks.

Segment Anything (via ``opengeos/segment-geospatial``) produces raster masks
over satellite or aerial imagery. This package turns those masks into *counted,
measured* features: connected-component labelling, per-object region properties,
area filtering, and pixel-to-metre/hectare conversions, plus an IoU helper for
validation. That quantification — building count, mean footprint in square
metres, total field area in hectares — is the runnable, tested contribution.

The numeric core (:mod:`samgeo_post.analytics`) and the demo
(:mod:`samgeo_post.demo`) depend only on numpy and the standard library, so they
import and test anywhere. The SAM segmentation wrapper
(:mod:`samgeo_post.segment`) and the vectoriser (:mod:`samgeo_post.vectorize`)
import their heavy dependencies lazily, inside functions, and are not imported
here or by the test suite.
"""

from __future__ import annotations

from samgeo_post.analytics import (
    area_hectares,
    count_objects,
    filter_by_area,
    label_components,
    mask_iou,
    pixels_to_area,
    region_props,
)
from samgeo_post.demo import run_demo

__all__ = [
    "label_components",
    "region_props",
    "count_objects",
    "filter_by_area",
    "pixels_to_area",
    "area_hectares",
    "mask_iou",
    "run_demo",
    "__version__",
]

__version__ = "0.1.0"
