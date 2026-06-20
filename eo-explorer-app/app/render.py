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
            "    pip install 'eo-monitor @ git+https://github.com/JosephMbuh/eo-monitor.git'\n\n"
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


def add_index_layer(map_obj, dataset, index: str, *, layer_name: str | None = None):
    """Compute ``index`` and add it to a leafmap ``Map`` as a coloured layer.

    Also adds a matching colour-bar legend. Returns the computed
    :class:`xarray.DataArray` for downstream use (e.g. stats display).
    """
    spec = INDEX_REGISTRY[index.upper()]
    data = compute_index(dataset, index)
    name = layer_name or f"{spec.name}"

    # Prefer leafmap's native xarray support so the GeoTIFF/tiles are handled
    # correctly; fall back to a coloured numpy image overlay if unavailable.
    try:
        map_obj.add_raster(
            data,
            colormap=spec.colormap,
            vmin=spec.vmin,
            vmax=spec.vmax,
            layer_name=name,
        )
    except Exception:  # noqa: BLE001 - older leafmap or non-georef array
        rgba = colorize(data, vmin=spec.vmin, vmax=spec.vmax, colormap=spec.colormap)
        bounds = _array_bounds(data)
        map_obj.add_image(rgba, bounds=bounds, layer_name=name)

    add_legend(map_obj, spec)
    return data


def add_legend(map_obj, spec: IndexSpec) -> None:
    """Add a colour-bar / legend describing ``spec`` to the map."""
    try:
        map_obj.add_colormap(
            cmap=spec.colormap,
            vmin=spec.vmin,
            vmax=spec.vmax,
            label=f"{spec.name} - {spec.description}",
        )
    except Exception:  # noqa: BLE001 - method name differs across leafmap versions
        try:
            map_obj.add_colorbar(
                colors=spec.colormap,
                vmin=spec.vmin,
                vmax=spec.vmax,
                caption=spec.name,
            )
        except Exception:  # noqa: BLE001 - legend is best-effort, never fatal
            pass


def _array_bounds(data) -> list[list[float]]:
    """Best-effort ``[[s, w], [n, e]]`` bounds for an xarray image overlay."""
    try:
        x = data["x"].values
        y = data["y"].values
        return [[float(y.min()), float(x.min())], [float(y.max()), float(x.max())]]
    except Exception:  # noqa: BLE001
        return [[-90.0, -180.0], [90.0, 180.0]]


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
