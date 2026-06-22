"""Build a lazy, Dask-backed xarray cube from STAC items via odc-stac.

Cloud masking uses the Sentinel-2 Scene Classification Layer (SCL). Library
code never calls ``.compute()`` — the cube stays lazy until export.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import xarray as xr
from odc.stac import load as odc_load

from eo_monitor.config import Config
from eo_monitor.indices import BAND_ALIASES, required_bands

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pystac import Item

logger = logging.getLogger(__name__)

# SCL classes to mask out: cloud shadows (3), cloud med/high prob (8, 9),
# thin cirrus (10) and snow/ice (11). See Sentinel-2 L2A product spec.
SCL_MASK_CLASSES: tuple[int, ...] = (3, 8, 9, 10, 11)

# Default chunking keeps tiles modest for Dask.
DEFAULT_CHUNKS: dict[str, int] = {"x": 1024, "y": 1024}


def _reverse_band_lookup() -> dict[str, str]:
    """Map asset key (e.g. 'B04') -> logical name (e.g. 'red')."""
    return {asset: logical for logical, asset in BAND_ALIASES.items()}


def load_cube(
    items: list[Item],
    config: Config,
    *,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Load items into a masked, Dask-backed :class:`xarray.Dataset`.

    The returned dataset has one variable per *logical* band (red/green/nir/swir,
    only those needed by the configured indices) plus the original SCL, with
    cloud/shadow/snow pixels set to NaN. Stays lazy.
    """
    chunks = chunks or DEFAULT_CHUNKS
    assets = required_bands(config.indices)
    # Always pull the Scene Classification Layer (asset "scl") for masking.
    bands = [*assets, "scl"]

    logger.info(
        "Loading cube: bands=%s res=%sm crs=%s groupby=%s",
        bands,
        config.resolution,
        config.crs,
        config.groupby,
    )
    ds = odc_load(
        items,
        bands=bands,
        resolution=config.resolution,
        crs=config.crs,
        bbox=config.aoi.bbox,
        groupby=config.groupby,
        chunks=chunks,
    )

    masked = apply_scl_mask(ds, scl_var="scl")

    # Rename asset keys to logical band names for downstream index math, and
    # normalise the SCL asset name to the upper-case "SCL" used elsewhere.
    rename = {
        asset: logical for asset, logical in _reverse_band_lookup().items() if asset in masked
    }
    if "scl" in masked:
        rename["scl"] = "SCL"
    masked = masked.rename(rename)
    return masked


def apply_scl_mask(ds: xr.Dataset, scl_var: str = "SCL") -> xr.Dataset:
    """Mask cloud/shadow/snow pixels using the SCL band; preserves laziness.

    Reflectance variables are cast to float and invalid pixels set to NaN. The
    SCL band itself is retained unmasked for traceability.
    """
    if scl_var not in ds:
        logger.warning("SCL band %r not present; skipping cloud mask.", scl_var)
        return ds

    scl = ds[scl_var]
    # True where the pixel is a *valid* (kept) observation.
    valid = ~scl.isin(list(SCL_MASK_CLASSES))

    out = ds.copy()
    for name in ds.data_vars:
        if name == scl_var:
            continue
        out[name] = ds[name].where(valid)
    return out
