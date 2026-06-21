"""Index computation + map-layer rendering for the EO Explorer app.

The spectral-index maths is **not reimplemented here**. Instead this module
imports the index functions from the sibling flagship package ``eo-monitor``
(``eo_monitor.indices``) so the portfolio projects visibly compose. If the
package is not installed we fall back to local, numerically-identical
implementations *only* so the smoke tests can exercise the pure colour-mapping
helpers; the production path always prefers ``eo_monitor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# eo-monitor reuse
# --------------------------------------------------------------------------- #

#: True when the real flagship package is importable.
EO_MONITOR_AVAILABLE = False

_EO_MONITOR_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised only when eo-monitor is installed
    from eo_monitor.indices import ndmi as _ndmi
    from eo_monitor.indices import ndvi as _ndvi
    from eo_monitor.indices import ndwi as _ndwi

    EO_MONITOR_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - we want any import failure here
    _EO_MONITOR_IMPORT_ERROR = exc

    # Local fallbacks (numerically identical normalised-difference formulas).
    # These let the colour-mapping helpers and smoke tests run without the
    # heavy dependency, but the app surfaces a clear message (see
    # ``require_eo_monitor``) so reviewers know the intent is real reuse.
    def _normalized_difference(a, b):
        import numpy as np

        a = np.asarray(a, dtype="float64")
        b = np.asarray(b, dtype="float64")
        denom = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (a - b) / denom
            out[denom == 0] = np.nan
        return out

    def _ndvi(red, nir):  # type: ignore[misc]
        return _normalized_difference(nir, red)

    def _ndwi(green, nir):  # type: ignore[misc]
        return _normalized_difference(green, nir)

    def _ndmi(nir, swir1):  # type: ignore[misc]
        return _normalized_difference(nir, swir1)


def require_eo_monitor() -> None:
    """Raise an actionable error if the flagship ``eo-monitor`` package is missing.

    Call this from the UI before computing an index so users get a specific
    install instruction rather than a silent fallback.
    """
    if not EO_MONITOR_AVAILABLE:
        raise ImportError(
            "This app reuses the index functions from the 'eo-monitor' package, "
            "which is not installed. Install it from the sibling repo, e.g.:\n\n"
            "    pip install -e ../eo-monitor\n\n"
            "or add the git dependency:\n\n"
            "    pip install 'eo-monitor @ git+https://github.com/mbongowo/Data-science-Portfolio.git@main#subdirectory=eo-monitor'\n\n"
            f"(original import error: {_EO_MONITOR_IMPORT_ERROR})"
        )


# --------------------------------------------------------------------------- #
# Index registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexSpec:
    """Describes one selectable spectral index."""

    name: str
    description: str
    func: Callable[..., Any]
    bands: tuple[str, ...]
    vmin: float
    vmax: float
    colormap: str


#: The indices the UI offers, wired to the (reused) eo-monitor functions.
INDEX_REGISTRY: dict[str, IndexSpec] = {
    "NDVI": IndexSpec(
        name="NDVI",
        description="Normalised Difference Vegetation Index (greenness).",
        func=_ndvi,
        bands=("red", "nir"),
        vmin=-0.2,
        vmax=0.9,
        colormap="RdYlGn",
    ),
    "NDWI": IndexSpec(
        name="NDWI",
        description="Normalised Difference Water Index (open water).",
        func=_ndwi,
        bands=("green", "nir"),
        vmin=-0.5,
        vmax=0.7,
        colormap="Blues",
    ),
    "NDMI": IndexSpec(
        name="NDMI",
        description="Normalised Difference Moisture Index (vegetation water content).",
        func=_ndmi,
        bands=("nir", "swir16"),
        vmin=-0.4,
        vmax=0.6,
        colormap="BrBG",
    ),
}


def list_indices() -> list[str]:
    """Return the selectable index names in display order."""
    return list(INDEX_REGISTRY)


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #


def compute_index(dataset, index: str):
    """Compute the named index from a loaded scene ``dataset``.

    ``dataset`` is an :class:`xarray.Dataset` whose variables are named after the
    Sentinel-2 assets in :data:`app.stac.INDEX_BANDS`. Returns an
    :class:`xarray.DataArray` of index values. Requires ``eo-monitor`` in the
    production path.
    """
    spec = INDEX_REGISTRY[index.upper()]
    require_eo_monitor()

    band_arrays = [dataset[b] for b in spec.bands]
    result = spec.func(*band_arrays)
    if hasattr(result, "rename"):
        result = result.rename(index.upper())
    return result


# --------------------------------------------------------------------------- #
# Colour mapping (pure-ish: needs numpy + matplotlib only)
# --------------------------------------------------------------------------- #


def normalize(values, vmin: float, vmax: float):
    """Scale ``values`` into ``[0, 1]`` against ``[vmin, vmax]`` (clipped).

    NaNs are preserved. Needs only numpy.
    """
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def colorize(values, *, vmin: float, vmax: float, colormap: str):
    """Map index ``values`` to an RGBA uint8 array using a matplotlib colormap.

    Returns an ``(H, W, 4)`` uint8 array with NaNs rendered transparent.
    """
    import matplotlib
    import numpy as np

    norm = normalize(values, vmin, vmax)
    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(np.asarray(norm))  # (H, W, 4) floats in [0, 1]

    nan_mask = ~np.isfinite(np.asarray(values, dtype="float64"))
    rgba[nan_mask, 3] = 0.0  # transparent where there is no data

    return (rgba * 255).astype("uint8")


def build_index_overlay(
    dataset, index: str, *, meta: dict | None = None, opacity: float = 0.8
) -> dict:
    """Compute ``index`` and package it as a folium image overlay.

    Returns a plain dict that ``app.main`` stores in session state and adds to a
    folium map as an :class:`folium.raster_layers.ImageOverlay`. The pixels are
    carried as a base64 PNG data URI, so the app needs no tile server and runs on
    a small host.

    The computed index is reprojected to EPSG:4326, the lon/lat space a folium
    overlay expects, and its rows are ordered north-to-south so the image lines up
    with the returned bounds.

    Parameters
    ----------
    dataset : xarray.Dataset
        A loaded scene with one variable per required band.
    index : str
        Spectral index name (NDVI / NDWI / NDMI).
    meta : dict, optional
        Scene metadata to carry alongside the overlay for display.
    opacity : float, optional
        Overlay opacity in ``[0, 1]``.

    Returns
    -------
    dict
        Keys: ``image_uri``, ``bounds`` (``[[south, west], [north, east]]``),
        ``name``, ``description``, ``vmin``, ``vmax``, ``colormap``, ``opacity``,
        ``stats``, and ``meta``.
    """
    import base64
    import io

    from PIL import Image

    spec = INDEX_REGISTRY[index.upper()]
    data = compute_index(dataset, index)

    data_latlon = _to_latlon(data)
    rgba = colorize(data_latlon, vmin=spec.vmin, vmax=spec.vmax, colormap=spec.colormap)

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    image_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "image_uri": image_uri,
        "bounds": _latlon_bounds(data_latlon),
        "name": spec.name,
        "description": spec.description,
        "vmin": spec.vmin,
        "vmax": spec.vmax,
        "colormap": spec.colormap,
        "opacity": float(opacity),
        "stats": index_stats(data),
        "meta": dict(meta or {}),
    }


def add_overlay_legend(folium_map, overlay: dict) -> None:
    """Add a branca colour-bar legend for a built overlay to ``folium_map``."""
    import branca.colormap as branca_cm
    import matplotlib
    import numpy as np

    cmap = matplotlib.colormaps[overlay["colormap"]]
    colours = [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0.0, 1.0, 9)]
    legend = branca_cm.LinearColormap(
        colours, vmin=overlay["vmin"], vmax=overlay["vmax"]
    )
    legend.caption = f"{overlay['name']} - {overlay['description']}"
    legend.add_to(folium_map)


def _to_latlon(data):
    """Reproject a computed index array to EPSG:4326 with north-up rows.

    ``load_scene`` loads imagery in EPSG:3857, so the index inherits that CRS. If
    the CRS was dropped during the index arithmetic it is set back before
    reprojecting. Rows are then ordered north-to-south so image row 0 is the
    northern edge, matching how a folium overlay reads its bounds.
    """
    import rioxarray  # noqa: F401 - registers the .rio accessor

    arr = data
    if arr.rio.crs is None:
        arr = arr.rio.write_crs("EPSG:3857")
    arr = arr.rio.reproject("EPSG:4326")
    if "y" in arr.dims and float(arr["y"][0]) < float(arr["y"][-1]):
        arr = arr.sortby("y", ascending=False)
    return arr


def _latlon_bounds(data_latlon) -> list[list[float]]:
    """Return ``[[south, west], [north, east]]`` lon/lat bounds for an overlay."""
    xs = data_latlon["x"].values
    ys = data_latlon["y"].values
    return [[float(ys.min()), float(xs.min())], [float(ys.max()), float(xs.max())]]


def index_stats(data) -> dict[str, float]:
    """Return simple summary statistics for a computed index array."""
    import numpy as np

    arr = np.asarray(data, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "valid_fraction": 0.0}
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "valid_fraction": float(finite.size / arr.size),
    }
