"""Polygonise labelled masks into georeferenced vector features.

This module converts an integer-labelled raster (one integer per object) into
vector polygons with per-polygon area, ready to write as GeoJSON or any vector
format. It is the bridge between the pure-numpy analytics core and a GIS.

All heavy geospatial dependencies (rasterio, shapely, geopandas) are imported
*inside* the functions, so importing this module is cheap and the test suite —
which never imports it — does not need the geospatial stack. Install the "full
pipeline" group from ``requirements.txt`` (or the conda env) to use it.
"""

from __future__ import annotations

from typing import Any


def masks_to_geojson(
    labeled: Any,
    transform: Any,
    crs: Any,
    pixel_size_m: float,
    out_path: str | None = None,
) -> Any:
    """Vectorise a labelled raster into polygons with per-polygon area.

    Parameters
    ----------
    labeled:
        2-D integer array (``0`` = background) as returned by
        :func:`samgeo_post.analytics.label_components`.
    transform:
        An ``affine.Affine`` raster transform mapping pixel -> map coordinates,
        as carried by the source GeoTIFF (``rasterio`` dataset ``.transform``).
    crs:
        The coordinate reference system of ``transform`` (anything
        ``geopandas`` accepts, e.g. ``"EPSG:32632"`` for Douala's UTM zone).
    pixel_size_m:
        Ground sample distance in metres; used to add an ``area_m2`` column
        computed from the pixel count, independent of the polygon geometry.
    out_path:
        If given, write the GeoDataFrame to this path as GeoJSON.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``label`` (int), ``area_px`` (int), ``area_m2`` (float), and
        ``geometry`` (the dissolved polygon for each label).

    Notes
    -----
    One label can map to several disjoint pixel groups only if the labelling
    allowed it; :func:`label_components` does not, so each label yields a single
    (possibly multipart) polygon here.
    """
    import geopandas as gpd
    import numpy as np
    from rasterio.features import shapes
    from shapely.geometry import shape

    from samgeo_post.analytics import pixels_to_area

    arr = np.asarray(labeled).astype("int32")
    records: list[dict[str, Any]] = []
    geoms = []
    # rasterio.features.shapes yields (geojson_geometry, value) for each
    # connected run of equal values; we keep the non-zero labels.
    for geom, value in shapes(arr, mask=arr != 0, transform=transform):
        lab = int(value)
        area_px = int(np.count_nonzero(arr == lab))
        records.append(
            {
                "label": lab,
                "area_px": area_px,
                "area_m2": pixels_to_area(area_px, pixel_size_m),
            }
        )
        geoms.append(shape(geom))

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=crs)
    if out_path is not None:
        gdf.to_file(out_path, driver="GeoJSON")
    return gdf
