"""Output: Cloud-Optimised GeoTIFF export and PNG quicklooks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

logger = logging.getLogger(__name__)


def write_cog(
    data: xr.DataArray,
    path: str | Path,
    *,
    overviews: bool = True,
    nodata: float = float("nan"),
) -> Path:
    """Write a 2-D (or single-band) DataArray to a Cloud-Optimised GeoTIFF.

    The array is materialised here (export is the one place compute is allowed),
    written to a temporary GeoTIFF, then translated in place to a valid COG with
    overviews via rio-cogeo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = data.squeeze()
    if "_FillValue" not in arr.attrs:
        arr = arr.rio.write_nodata(nodata, inplace=False)

    tmp = path.with_suffix(".tmp.tif")
    # rioxarray drives the GDAL write; this triggers Dask compute for the tile.
    arr.rio.to_raster(tmp, driver="GTiff")

    profile = cog_profiles.get("deflate")
    overview_levels = 5 if overviews else 0
    cog_translate(
        tmp,
        path,
        profile,
        overview_level=overview_levels,
        overview_resampling="average",
        in_memory=False,
        quiet=True,
    )
    tmp.unlink(missing_ok=True)
    logger.info("Wrote COG: %s", path)
    return path


def write_quicklook(
    data: xr.DataArray,
    path: str | Path,
    *,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> Path:
    """Render a PNG quicklook of a 2-D DataArray with matplotlib.

    Uses a non-interactive backend so it works headless (CI / Docker).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data.squeeze().values, dtype="float64")
    if vmin is None or vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2, 98])
        else:
            lo, hi = -1.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote quicklook: %s", path)
    return path
