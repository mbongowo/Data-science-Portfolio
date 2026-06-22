"""Construction and comparison of spatial weights matrices.

A spatial weights matrix ``W`` encodes which areal units are "neighbours" and
how strongly. The choice of ``W`` is a modelling decision, not a detail: every
downstream ESDA statistic (Moran's I, LISA, Gi*) is conditional on it. This
module builds the three most common families and reports diagnostics so the
choice can be justified rather than assumed.

* **Queen contiguity** — units sharing an edge or a vertex are neighbours.
  Good default for irregular polygons (counties). Sensitive to topology errors
  and to "islands" (units with no contiguous neighbour).
* **Distance band** — neighbours within a fixed radius. Requires a sensible
  threshold (often the minimum distance that guarantees no islands) and is
  sensitive to the modifiable areal unit problem and to scale.
* **K-nearest neighbours (KNN)** — each unit always has exactly ``k``
  neighbours. Guarantees no islands by construction, but imposes an asymmetric,
  fixed-cardinality structure that may not reflect real adjacency.

All builders optionally **row-standardise** (``transform="r"``) so that each
row sums to one, which makes the spatial lag a local average — the usual choice
for ESDA. Islands are handled *explicitly*: by default they raise, because a
silent island quietly drops observations from the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd


class IslandError(ValueError):
    """Raised when a weights matrix contains disconnected units (islands)."""


@dataclass
class WeightsDiagnostics:
    """Summary diagnostics for a spatial weights matrix."""

    kind: str
    n: int
    n_islands: int
    island_ids: list[Any]
    mean_neighbors: float
    min_neighbors: int
    max_neighbors: int
    pct_nonzero: float
    transform: str

    def as_dict(self) -> dict[str, Any]:
        """Return the diagnostics as a plain dict (handy for logging/JSON)."""
        return {
            "kind": self.kind,
            "n": self.n,
            "n_islands": self.n_islands,
            "island_ids": self.island_ids,
            "mean_neighbors": self.mean_neighbors,
            "min_neighbors": self.min_neighbors,
            "max_neighbors": self.max_neighbors,
            "pct_nonzero": self.pct_nonzero,
            "transform": self.transform,
        }


def _check_islands(w: Any, *, on_islands: Literal["raise", "warn", "ignore"]) -> None:
    """Inspect ``w.islands`` and react according to ``on_islands``."""
    islands = list(getattr(w, "islands", []))
    if not islands:
        return
    msg = (
        f"Weights matrix has {len(islands)} island(s) with no neighbours: "
        f"{islands}. These units are excluded from spatial statistics. "
        "Consider a distance band large enough to connect them, KNN, or "
        "dropping them deliberately."
    )
    if on_islands == "raise":
        raise IslandError(msg)
    if on_islands == "warn":
        import warnings

        warnings.warn(msg, stacklevel=2)
    # "ignore": do nothing


def _maybe_row_standardize(w: Any, row_standardize: bool) -> Any:
    if row_standardize:
        w.transform = "r"
    return w


def queen_contiguity(
    gdf: "gpd.GeoDataFrame",
    *,
    row_standardize: bool = True,
    on_islands: Literal["raise", "warn", "ignore"] = "raise",
) -> Any:
    """Build a Queen contiguity weights matrix from polygon geometries.

    Parameters
    ----------
    gdf:
        GeoDataFrame of polygons (the areal units).
    row_standardize:
        If True, set ``transform="r"`` so each row sums to one.
    on_islands:
        How to react to disconnected units. ``"raise"`` (default), ``"warn"``,
        or ``"ignore"``.
    """
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf, use_index=True)
    _check_islands(w, on_islands=on_islands)
    return _maybe_row_standardize(w, row_standardize)


def distance_band(
    gdf: "gpd.GeoDataFrame",
    *,
    threshold: float | None = None,
    row_standardize: bool = True,
    on_islands: Literal["raise", "warn", "ignore"] = "raise",
) -> Any:
    """Build a distance-band weights matrix from unit centroids.

    Parameters
    ----------
    gdf:
        GeoDataFrame in a *projected* CRS (so distances are in metres, not
        degrees). The centroid of each geometry is used as its location.
    threshold:
        Band radius in CRS units. If ``None``, the minimum threshold that
        guarantees every unit has at least one neighbour is used
        (``min_threshold_distance``), which avoids islands by construction.
    row_standardize, on_islands:
        See :func:`queen_contiguity`.
    """
    from libpysal.weights import DistanceBand, min_threshold_distance
    from libpysal.weights.util import get_points_array_from_shapefile  # noqa: F401

    points = [(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry]
    if threshold is None:
        threshold = float(min_threshold_distance(points))

    w = DistanceBand(points, threshold=threshold, ids=list(gdf.index))
    _check_islands(w, on_islands=on_islands)
    return _maybe_row_standardize(w, row_standardize)


def knn(
    gdf: "gpd.GeoDataFrame",
    *,
    k: int = 8,
    row_standardize: bool = True,
) -> Any:
    """Build a K-nearest-neighbours weights matrix from unit centroids.

    KNN cannot produce islands (every unit gets exactly ``k`` neighbours), so
    there is no ``on_islands`` argument. Note that the resulting relation is
    generally **asymmetric**.
    """
    from libpysal.weights import KNN

    w = KNN.from_dataframe(gdf, k=k)
    return _maybe_row_standardize(w, row_standardize)


def diagnostics(w: Any) -> WeightsDiagnostics:
    """Compute neighbour-count diagnostics for a weights matrix.

    Reports the mean / min / max number of neighbours, the count and ids of
    islands, the sparsity (% of non-zero cells) and the current transform.
    """
    cardinalities = list(w.cardinalities.values())
    n = w.n
    islands = list(getattr(w, "islands", []))
    nonzero = sum(cardinalities)
    pct_nonzero = 100.0 * nonzero / (n * n) if n else 0.0
    return WeightsDiagnostics(
        kind=type(w).__name__,
        n=n,
        n_islands=len(islands),
        island_ids=islands,
        mean_neighbors=float(sum(cardinalities) / n) if n else 0.0,
        min_neighbors=int(min(cardinalities)) if cardinalities else 0,
        max_neighbors=int(max(cardinalities)) if cardinalities else 0,
        pct_nonzero=pct_nonzero,
        transform=str(w.transform),
    )


def compare_weights(weights: dict[str, Any]) -> list[WeightsDiagnostics]:
    """Compute :func:`diagnostics` for several named weights matrices.

    Parameters
    ----------
    weights:
        Mapping of label -> ``W`` object, e.g.
        ``{"queen": w_queen, "knn8": w_knn}``.

    Returns
    -------
    list of WeightsDiagnostics
        One entry per input, with ``kind`` overwritten by the supplied label so
        the report is easy to read.
    """
    out: list[WeightsDiagnostics] = []
    for label, w in weights.items():
        d = diagnostics(w)
        d.kind = label
        out.append(d)
    return out
