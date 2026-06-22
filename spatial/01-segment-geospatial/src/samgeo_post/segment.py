"""Run Segment Anything over a basemap AOI (heavy; GPU / Colab recommended).

This is a thin wrapper around ``opengeos/segment-geospatial`` (``samgeo``). It
pulls basemap tiles for an area of interest and runs SAM's automatic mask
generator to produce a labelled mask GeoTIFF that the pure-numpy analytics core
then quantifies.

Credit: this project is inspired by and builds on
``opengeos/segment-geospatial`` (https://github.com/opengeos/segment-geospatial).
The contribution here is the downstream quantification, not the segmentation.

Heavy: this needs ``samgeo`` + ``torch`` + ``leafmap`` and a CUDA GPU to be
practical; the SAM checkpoint is large. Run it in Google Colab with a GPU
runtime, or on a local GPU box. Everything is imported lazily inside the
function so that importing this module — and the entire test suite — needs none
of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def segment_basemap(
    aoi_bbox: tuple[float, float, float, float],
    zoom: int = 19,
    out_tif: str | Path = "outputs/douala_masks.tif",
    source: str = "Satellite",
    checkpoint: str | None = None,
) -> str:
    """Download basemap tiles for an AOI and run SAM auto-segmentation.

    Parameters
    ----------
    aoi_bbox:
        ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326. For Douala see
        ``config/aoi.yaml``.
    zoom:
        Basemap tile zoom level. Higher means finer resolution and more tiles;
        18-20 is typical for building-scale segmentation.
    out_tif:
        Path for the written mask GeoTIFF (integer object labels).
    source:
        Basemap source understood by ``samgeo.tms_to_geotiff`` (e.g.
        ``"Satellite"``).
    checkpoint:
        Optional path to a SAM checkpoint; if ``None``, ``samgeo`` downloads the
        default model.

    Returns
    -------
    str
        Path to the written labelled mask GeoTIFF.

    Notes
    -----
    Needs a GPU to be practical. Imports are deferred so this module stays
    importable without the deep-learning stack.
    """
    from samgeo import SamGeo, tms_to_geotiff

    out_tif = str(out_tif)
    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)

    # 1. Pull basemap imagery for the AOI into a GeoTIFF.
    image_tif = str(Path(out_tif).with_name("aoi_image.tif"))
    min_lon, min_lat, max_lon, max_lat = aoi_bbox
    tms_to_geotiff(
        output=image_tif,
        bbox=[min_lon, min_lat, max_lon, max_lat],
        zoom=zoom,
        source=source,
        overwrite=True,
    )

    # 2. Run SAM automatic mask generation, writing an integer-labelled GeoTIFF.
    sam_kwargs: dict[str, Any] = {"model_type": "vit_h", "sam_kwargs": None}
    if checkpoint is not None:
        sam_kwargs["checkpoint"] = checkpoint
    sam = SamGeo(**sam_kwargs)
    sam.generate(image_tif, output=out_tif, foreground=True, unique=True)
    return out_tif
