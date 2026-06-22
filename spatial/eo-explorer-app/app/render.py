"""Index computation + map-layer rendering for the EO Explorer app.

The spectral-index maths is **not reimplemented here**. Instead this module
imports the index functions from the sibling flagship package ``eo-monitor``
(``eo_monitor.indices``) so the portfolio projects visibly compose. If the
package is not installed we fall back to local, numerically-identical
implementations *only* so the smoke tests can exercise the pure colour-mapping
helpers; the production path always prefers ``eo_monitor``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# eo-monitor reuse
# --------------------------------------------------------------------------- #

#: True when the real flagship package is importable.
EO_MONITOR_AVAILABLE = False

_EO_MONITOR_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised only when eo-monitor is installed
    from eo_monitor import indices as _eo_idx

    _ndvi = _eo_idx.ndvi
    _ndwi = _eo_idx.ndwi
    _ndmi = _eo_idx.ndmi

    EO_MONITOR_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - we want any import failure here
    _eo_idx = None
    _EO_MONITOR_IMPORT_ERROR = exc

    # Local fallbacks (numerically identical normalised-difference formulas).
    # These let the colour-mapping helpers and smoke tests run without the
    # heavy dependency, but the app surfaces a clear message (see
    # ``require_eo_monitor``) so reviewers know the intent is real reuse. Only
    # the smoke-tested normalised-difference indices need a fallback; the wider
    # catalogue resolves from eo-monitor in the production path.
    def _normalized_difference(a, b):
        import numpy as np

        a = np.asarray(a, dtype="float64")
        b = np.asarray(b, dtype="float64")
        denom = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (a - b) / denom
            out[denom == 0] = np.nan
        return out

    def _ndvi(nir, red):  # type: ignore[misc]
        return _normalized_difference(nir, red)

    def _ndwi(green, nir):  # type: ignore[misc]
        return _normalized_difference(green, nir)

    def _ndmi(nir, swir1):  # type: ignore[misc]
        return _normalized_difference(nir, swir1)


def _eo_func(name: str, fallback: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """Return the eo-monitor index function ``name``, or a local fallback.

    The production path always prefers the reused ``eo_monitor.indices``
    function. When eo-monitor is not installed we use ``fallback`` if one is
    given (the three normalised-difference indices the smoke tests exercise),
    otherwise a stub that raises a clear error the moment it is called -- the UI
    already blocks compute via :func:`require_eo_monitor`, so this stub is only a
    safety net and never runs in the tested path.
    """
    if _eo_idx is not None:
        func = getattr(_eo_idx, name, None)
        if func is not None:
            return func
        # eo-monitor is installed but older than this app and does not define
        # this index. Degrade gracefully (fall through) instead of raising at
        # import time, which would take the whole app down. A stale deployment
        # cache is the usual cause.

    if fallback is not None:
        return fallback

    def _missing(*_args, **_kwargs):
        require_eo_monitor()
        raise ImportError(
            f"eo-monitor index {name!r} is unavailable. The installed eo-monitor "
            "is older than this app; reinstall it from this repository."
        )

    return _missing


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
            "    pip install 'eo-monitor @ git+https://github.com/mbongowo/Data-science-Portfolio.git@main#subdirectory=spatial/eo-monitor'\n\n"
            f"(original import error: {_EO_MONITOR_IMPORT_ERROR})"
        )


# --------------------------------------------------------------------------- #
# Index registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexSpec:
    """Describes one selectable spectral index.

    ``bands`` lists the Sentinel-2 L2A asset names in the exact order the
    ``func`` takes them positionally (``compute_index`` calls
    ``func(*[dataset[b] for b in spec.bands])``), so the order here must match
    the reused eo-monitor signature. ``category`` groups the index in the UI.
    """

    name: str
    description: str
    func: Callable[..., Any]
    bands: tuple[str, ...]
    vmin: float
    vmax: float
    colormap: str
    category: str


def _spec(name, description, eo_name, bands, vmin, vmax, colormap, category, fallback=None):
    """Build an :class:`IndexSpec` wired to the reused eo-monitor function."""
    return IndexSpec(
        name=name,
        description=description,
        func=_eo_func(eo_name, fallback),
        bands=bands,
        vmin=vmin,
        vmax=vmax,
        colormap=colormap,
        category=category,
    )


# Sentinel-2 L2A (Earth Search) asset names used below:
#   blue, green, red, rededge1, nir, swir16 (SWIR1), swir22 (SWIR2).
# Each ``bands`` tuple is ordered to match the reused eo_monitor.indices
# function signature. Indices with additive constants (EVI, SAVI, MSAVI, AWEI,
# BAI, LAI) assume surface reflectance in [0, 1]; stac.load_scene scales the
# bands accordingly before compute_index runs.
INDEX_REGISTRY: dict[str, IndexSpec] = {
    # ----------------------------- Vegetation ----------------------------- #
    "NDVI": _spec(
        "NDVI",
        "Normalised Difference Vegetation Index (greenness).",
        "ndvi",
        ("nir", "red"),
        -0.2,
        0.9,
        "RdYlGn",
        "Vegetation",
        fallback=_ndvi,
    ),
    "EVI": _spec(
        "EVI",
        "Enhanced Vegetation Index (aerosol-corrected greenness).",
        "evi",
        ("nir", "red", "blue"),
        -0.2,
        1.0,
        "RdYlGn",
        "Vegetation",
    ),
    "EVI2": _spec(
        "EVI2",
        "Two-band Enhanced Vegetation Index (no blue band).",
        "evi2",
        ("nir", "red"),
        -0.2,
        1.0,
        "RdYlGn",
        "Vegetation",
    ),
    "SAVI": _spec(
        "SAVI",
        "Soil-Adjusted Vegetation Index (L=0.5).",
        "savi",
        ("nir", "red"),
        -0.2,
        0.9,
        "RdYlGn",
        "Vegetation",
    ),
    "MSAVI": _spec(
        "MSAVI",
        "Modified Soil-Adjusted Vegetation Index (self-adjusting L).",
        "msavi",
        ("nir", "red"),
        -0.2,
        0.9,
        "RdYlGn",
        "Vegetation",
    ),
    "GNDVI": _spec(
        "GNDVI",
        "Green NDVI (chlorophyll-sensitive greenness).",
        "gndvi",
        ("nir", "green"),
        -0.2,
        0.9,
        "RdYlGn",
        "Vegetation",
    ),
    "ARVI": _spec(
        "ARVI",
        "Atmospherically Resistant Vegetation Index.",
        "arvi",
        ("nir", "red", "blue"),
        -0.2,
        0.9,
        "RdYlGn",
        "Vegetation",
    ),
    "NDRE": _spec(
        "NDRE",
        "Normalised Difference Red-Edge (canopy chlorophyll).",
        "ndre",
        ("nir", "rededge1"),
        -0.2,
        0.8,
        "RdYlGn",
        "Vegetation",
    ),
    "VARI": _spec(
        "VARI",
        "Visible Atmospherically Resistant Index (RGB greenness).",
        "vari",
        ("green", "red", "blue"),
        -0.5,
        0.5,
        "RdYlGn",
        "Vegetation",
    ),
    "RVI": _spec(
        "RVI",
        "Ratio Vegetation Index = NIR / Red.",
        "rvi",
        ("nir", "red"),
        0.0,
        10.0,
        "YlGn",
        "Vegetation",
    ),
    "DVI": _spec(
        "DVI",
        "Difference Vegetation Index = NIR - Red.",
        "dvi",
        ("nir", "red"),
        -0.1,
        0.6,
        "YlGn",
        "Vegetation",
    ),
    "CIGREEN": _spec(
        "CIgreen",
        "Chlorophyll Index - green = NIR / Green - 1.",
        "ci_green",
        ("nir", "green"),
        0.0,
        8.0,
        "YlGn",
        "Vegetation",
    ),
    "CIREDEDGE": _spec(
        "CIrededge",
        "Chlorophyll Index - red-edge = NIR / RE - 1.",
        "ci_rededge",
        ("nir", "rededge1"),
        0.0,
        6.0,
        "YlGn",
        "Vegetation",
    ),
    "MCARI": _spec(
        "MCARI",
        "Modified Chlorophyll Absorption in Reflectance Index.",
        "mcari",
        ("rededge1", "red", "green"),
        0.0,
        1.5,
        "YlGn",
        "Vegetation",
    ),
    "TCARI": _spec(
        "TCARI",
        "Transformed Chlorophyll Absorption in Reflectance Index.",
        "tcari",
        ("rededge1", "red", "green"),
        0.0,
        1.5,
        "YlGn",
        "Vegetation",
    ),
    "LAI": _spec(
        "LAI",
        "Leaf Area Index (approximate empirical relation from EVI).",
        "lai",
        ("nir", "red", "blue"),
        0.0,
        6.0,
        "YlGn",
        "Vegetation",
    ),
    # ------------------------------- Water -------------------------------- #
    "NDWI": _spec(
        "NDWI",
        "Normalised Difference Water Index (open water).",
        "ndwi",
        ("green", "nir"),
        -0.5,
        0.7,
        "Blues",
        "Water",
        fallback=_ndwi,
    ),
    "MNDWI": _spec(
        "MNDWI",
        "Modified NDWI (Green/SWIR1, suppresses built-up).",
        "mndwi",
        ("green", "swir16"),
        -0.5,
        0.8,
        "Blues",
        "Water",
    ),
    "NDMI": _spec(
        "NDMI",
        "Normalised Difference Moisture Index (canopy water).",
        "ndmi",
        ("nir", "swir16"),
        -0.4,
        0.6,
        "BrBG",
        "Water",
        fallback=_ndmi,
    ),
    "AWEI": _spec(
        "AWEI",
        "Automated Water Extraction Index (no-shadow form).",
        "awei",
        ("green", "nir", "swir16", "swir22"),
        -2.0,
        2.0,
        "Blues",
        "Water",
    ),
    "NDII": _spec(
        "NDII",
        "Normalised Difference Infrared Index (= NDMI formula).",
        "ndii",
        ("nir", "swir16"),
        -0.4,
        0.6,
        "BrBG",
        "Water",
    ),
    # ------------------------------- Soil --------------------------------- #
    "BSI": _spec(
        "BSI",
        "Bare Soil Index.",
        "bsi",
        ("swir16", "red", "nir", "blue"),
        -0.5,
        0.5,
        "YlOrBr",
        "Soil",
    ),
    "SI": _spec(
        "SI",
        "Soil Salinity Index = sqrt(Green*Red) (one common form).",
        "salinity_index",
        ("green", "red"),
        0.0,
        0.5,
        "YlOrBr",
        "Soil",
    ),
    "IRONOXIDE": _spec(
        "IronOxide",
        "Iron Oxide ratio = Red / Blue.",
        "iron_oxide",
        ("red", "blue"),
        0.5,
        3.0,
        "OrRd",
        "Soil",
    ),
    "CLAYMINERALS": _spec(
        "ClayMinerals",
        "Clay Minerals ratio = SWIR1 / SWIR2.",
        "clay_minerals",
        ("swir16", "swir22"),
        0.8,
        2.0,
        "OrRd",
        "Soil",
    ),
    "FERROUSMINERALS": _spec(
        "FerrousMinerals",
        "Ferrous Minerals ratio = SWIR1 / NIR.",
        "ferrous_minerals",
        ("swir16", "nir"),
        0.3,
        1.5,
        "OrRd",
        "Soil",
    ),
    # ------------------------------ Built-up ------------------------------ #
    "NDBI": _spec(
        "NDBI",
        "Normalised Difference Built-up Index.",
        "ndbi",
        ("swir16", "nir"),
        -0.5,
        0.5,
        "PuRd",
        "Built-up",
    ),
    "UI": _spec(
        "UI",
        "Urban Index = (SWIR2 - NIR) / (SWIR2 + NIR).",
        "ui",
        ("swir22", "nir"),
        -0.5,
        0.5,
        "PuRd",
        "Built-up",
    ),
    "IBI": _spec(
        "IBI",
        "Index-Based Built-up Index (combines NDBI, SAVI, MNDWI).",
        "ibi",
        ("swir16", "nir", "red", "green"),
        -0.5,
        0.5,
        "PuRd",
        "Built-up",
    ),
    # ------------------------------- Snow --------------------------------- #
    "NDSI": _spec(
        "NDSI",
        "Normalised Difference Snow Index.",
        "ndsi",
        ("green", "swir16"),
        -0.5,
        1.0,
        "PuBu",
        "Snow",
    ),
    "NDGI": _spec(
        "NDGI",
        "Normalised Difference Glacier Index (green/red variant).",
        "ndgi",
        ("green", "red"),
        -0.5,
        0.5,
        "PuBu",
        "Snow",
    ),
    # ------------------------------- Fire --------------------------------- #
    "NBR": _spec(
        "NBR",
        "Normalised Burn Ratio (burn severity).",
        "nbr",
        ("nir", "swir22"),
        -0.5,
        1.0,
        "RdYlGn",
        "Fire",
    ),
    "NBR2": _spec(
        "NBR2",
        "Normalised Burn Ratio 2 (SWIR1/SWIR2).",
        "nbr2",
        ("swir16", "swir22"),
        -0.5,
        0.5,
        "RdYlGn",
        "Fire",
    ),
    "BAI": _spec(
        "BAI",
        "Burned Area Index (highlights charcoal-dark scars).",
        "bai",
        ("red", "nir"),
        0.0,
        500.0,
        "inferno",
        "Fire",
    ),
}


def list_indices() -> list[str]:
    """Return the selectable index names in display order."""
    return list(INDEX_REGISTRY)


def list_indices_by_category() -> dict[str, list[str]]:
    """Return a mapping of category -> list of index names, in registry order."""
    out: dict[str, list[str]] = {}
    for name, spec in INDEX_REGISTRY.items():
        out.setdefault(spec.category, []).append(name)
    return out


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
    legend = branca_cm.LinearColormap(colours, vmin=overlay["vmin"], vmax=overlay["vmax"])
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


def percentile_stretch(values, lo: float = 2.0, hi: float = 98.0):
    """Return robust ``(vmin, vmax)`` contrast bounds, ignoring NaN.

    A linear stretch between the global min and max is fragile: a few extreme
    pixels squash everything else into a narrow band of colour. Clipping to a
    low and a high percentile (2 and 98 by default) discards those outliers and
    gives a stretch that uses the colour ramp on the bulk of the data.

    Parameters
    ----------
    values : array-like
        Index values; NaNs are ignored.
    lo, hi : float, optional
        Lower and upper percentiles in ``[0, 100]`` with ``lo < hi``.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)``. If no finite values are present both are NaN; if every
        finite value is identical both equal that value.

    Raises
    ------
    ValueError
        If ``lo`` is not strictly less than ``hi``.
    """
    import numpy as np

    if lo >= hi:
        raise ValueError("lo must be strictly less than hi")
    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    vmin, vmax = np.percentile(finite, [lo, hi])
    return (float(vmin), float(vmax))


def histogram(values, bins):
    """Return ``(counts, edges)`` for a histogram of ``values``, ignoring NaN.

    A thin wrapper over :func:`numpy.histogram` that first drops non-finite
    values, so NaNs and infinities never land in a bin or skew the range.

    Parameters
    ----------
    values : array-like
        Index values; NaNs and infinities are ignored.
    bins : int or sequence of float
        Bin count or explicit bin edges, passed straight to
        :func:`numpy.histogram`.

    Returns
    -------
    tuple of numpy.ndarray
        ``counts`` (length ``len(edges) - 1``) and ``edges``. When there are no
        finite values the counts are all zero over the default ``[0, 1]`` range.
    """
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.histogram(np.array([]), bins=bins, range=(0.0, 1.0))
    counts, edges = np.histogram(finite, bins=bins)
    return counts, edges


def downsample(arr, max_dim: int):
    """Strided nearest-neighbour downsample so the longest side <= ``max_dim``.

    A cheap preview helper: it takes every ``step``-th row and column, where
    ``step`` is the smallest integer that brings the longer axis to or below
    ``max_dim``. No interpolation or averaging, so it is fast and introduces no
    new values (NaNs stay NaN). An array already within the limit is returned
    unchanged.

    Parameters
    ----------
    arr : array-like
        A 2-D array (H, W).
    max_dim : int
        Maximum allowed length of the longer side, in pixels. Must be positive.

    Returns
    -------
    numpy.ndarray
        The strided view's contents. The longer side is ``<= max_dim``.

    Raises
    ------
    ValueError
        If ``max_dim`` is not positive or ``arr`` is not 2-D.
    """
    import numpy as np

    if max_dim <= 0:
        raise ValueError("max_dim must be positive")
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("downsample expects a 2-D array")
    longest = max(a.shape)
    if longest <= max_dim:
        return a
    step = -(-longest // max_dim)  # ceil division
    return a[::step, ::step]


def index_stats(data) -> dict[str, float]:
    """Return simple summary statistics for a computed index array."""
    import numpy as np

    arr = np.asarray(data, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "valid_fraction": 0.0,
        }
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "valid_fraction": float(finite.size / arr.size),
    }
