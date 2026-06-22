"""Compute nearest-facility travel time for every demand point.

Pipeline role: run a multi-source shortest-path search from all facility nodes
over the ``travel_time`` edge weight, then assign every demand point the travel
time to its closest facility. Build the demand surface as H3 cells or a regular
population grid clipped to the study area.

:func:`nearest_facility_times` is a pure-Python multi-source Dijkstra mirroring
``networkx.multi_source_dijkstra`` semantics, so the core routing logic is
unit-testable on a tiny synthetic graph without any geospatial dependency.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "sources.yaml"


def nearest_facility_times(
    adjacency: dict[Hashable, list[tuple[Hashable, float]]],
    sources: list[Hashable],
) -> dict[Hashable, float]:
    """Multi-source Dijkstra returning the minimum-cost distance to any source.

    Args:
        adjacency: mapping ``node -> [(neighbour, weight), ...]`` with
            non-negative weights (here, edge ``travel_time`` in seconds).
        sources: facility nodes, each seeded at cost 0.

    Returns:
        ``node -> cost`` for every node reachable from any source. Unreachable
        nodes are omitted. This matches ``networkx.multi_source_dijkstra``'s
        distance output and is the engine behind the GeoDataFrame builder.
    """
    dist: dict[Hashable, float] = {}
    heap: list[tuple[float, int, Hashable]] = []
    counter = 0
    for src in sources:
        if src not in dist:
            dist[src] = 0.0
            heapq.heappush(heap, (0.0, counter, src))
            counter += 1

    while heap:
        cost, _, node = heapq.heappop(heap)
        if cost > dist.get(node, math.inf):
            continue
        for neighbour, weight in adjacency.get(node, ()):  # type: ignore[arg-type]
            if weight < 0:
                raise ValueError("negative edge weights are not allowed")
            new_cost = cost + weight
            if new_cost < dist.get(neighbour, math.inf):
                dist[neighbour] = new_cost
                heapq.heappush(heap, (new_cost, counter, neighbour))
                counter += 1
    return dist


def assign_nearest_source(
    adjacency: dict[Hashable, list[tuple[Hashable, float]]],
    sources: list[Hashable],
) -> tuple[dict[Hashable, float], dict[Hashable, Hashable]]:
    """Multi-source Dijkstra returning both the cost and the nearest source.

    Extends :func:`nearest_facility_times` to also track *which* source each node
    is closest to. The source label propagates along the relaxed edge, so every
    reachable node carries the seed it was reached from on the shortest path.
    This is what :func:`access.metrics.facility_load` needs to attribute demand
    to facilities.

    Args:
        adjacency: mapping ``node -> [(neighbour, weight), ...]`` with
            non-negative weights.
        sources: facility nodes, each seeded at cost 0.

    Returns:
        ``(dist, nearest)`` where ``dist`` maps ``node -> cost`` (as in
        :func:`nearest_facility_times`) and ``nearest`` maps ``node -> source``
        for every reachable node. Unreachable nodes appear in neither dict. Ties
        resolve to the source pushed first (stable insertion order of ``sources``).
    """
    dist: dict[Hashable, float] = {}
    nearest: dict[Hashable, Hashable] = {}
    heap: list[tuple[float, int, Hashable]] = []
    counter = 0
    for src in sources:
        if src not in dist:
            dist[src] = 0.0
            nearest[src] = src
            heapq.heappush(heap, (0.0, counter, src))
            counter += 1

    while heap:
        cost, _, node = heapq.heappop(heap)
        if cost > dist.get(node, math.inf):
            continue
        for neighbour, weight in adjacency.get(node, ()):  # type: ignore[arg-type]
            if weight < 0:
                raise ValueError("negative edge weights are not allowed")
            new_cost = cost + weight
            if new_cost < dist.get(neighbour, math.inf):
                dist[neighbour] = new_cost
                nearest[neighbour] = nearest[node]
                heapq.heappush(heap, (new_cost, counter, neighbour))
                counter += 1
    return dist, nearest


def graph_to_adjacency(
    graph, weight: str = "travel_time"
) -> dict[Hashable, list[tuple[Hashable, float]]]:
    """Convert a networkx graph to a plain adjacency dict on ``weight``."""
    adjacency: dict[Hashable, list[tuple[Hashable, float]]] = {n: [] for n in graph.nodes}
    for u, v, data in graph.edges(data=True):
        adjacency[u].append((v, float(data.get(weight, 0.0))))
    return adjacency


def seconds_to_minutes(seconds: float) -> float:
    """Convert a travel time in seconds to minutes, propagating unreachable.

    A non-finite input (NaN or inf, marking a demand cell that cannot reach any
    facility) returns NaN rather than a spurious finite value.

    Parameters
    ----------
    seconds : float
        Travel time in seconds, or NaN/inf for an unreachable cell.

    Returns
    -------
    float
        Travel time in minutes, or NaN when the input is not finite.
    """
    if not math.isfinite(seconds):
        return math.nan
    return seconds / 60.0


def nearest_times_to_minutes(
    node_ids: Sequence[Hashable],
    times_seconds: dict[Hashable, float],
) -> list[float]:
    """Map snapped node ids to nearest-facility travel time in minutes.

    For each node id this looks up the multi-source shortest-path cost (in
    seconds) and converts it to minutes. A node missing from ``times_seconds``
    (unreachable from every facility) maps to NaN. Pure Python; this is the part
    of :func:`assign_access_to_demand` that does not need any geospatial library.

    Parameters
    ----------
    node_ids : sequence of hashable
        Graph node id per demand cell, in demand-cell order.
    times_seconds : dict of hashable to float
        Output of :func:`nearest_facility_times` /
        ``networkx.multi_source_dijkstra_path_length``.

    Returns
    -------
    list of float
        Travel time in minutes per demand cell, NaN where unreachable.
    """
    return [seconds_to_minutes(times_seconds.get(n, math.nan)) for n in node_ids]


def compute_access_times(graph, source_nodes: list[Hashable], weight: str = "travel_time"):
    """Return ``node -> travel_time_seconds`` to the nearest source over ``graph``.

    Prefers ``networkx.multi_source_dijkstra_path_length`` when available; falls
    back to the pure-Python engine otherwise (keeps results identical).
    """
    try:
        import networkx as nx

        return nx.multi_source_dijkstra_path_length(graph, set(source_nodes), weight=weight)
    except ImportError:  # pragma: no cover - exercised only without networkx
        return nearest_facility_times(graph_to_adjacency(graph, weight), source_nodes)


def build_demand_surface(study_area, cfg: dict[str, Any]):
    """Build a demand-point GeoDataFrame (H3 cell centroids or a regular grid).

    Requires geopandas/shapely; clips to ``study_area`` (a GeoDataFrame in the
    geographic CRS).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    demand_cfg = cfg["demand"]
    geo_crs = cfg["study_area"]["geographic_crs"]
    proj_crs = cfg["study_area"]["projected_crs"]

    area = study_area.to_crs(geo_crs)
    unioned = area.union_all() if hasattr(area, "union_all") else area.unary_union

    points: list[Point] = []
    if demand_cfg["type"] == "h3":
        import h3

        res = int(demand_cfg["h3_resolution"])
        geojson = {
            "type": "Polygon",
            "coordinates": [list(unioned.exterior.coords)],
        }
        try:  # h3 v4 API
            cells = h3.polygon_to_cells(h3.geo_to_polygon(geojson), res)  # type: ignore[attr-defined]
        except AttributeError:  # h3 v3 API
            cells = h3.polyfill(geojson, res, geo_json_conformant=True)
        for cell in cells:
            try:
                lat, lon = h3.cell_to_latlng(cell)  # v4
            except AttributeError:
                lat, lon = h3.h3_to_geo(cell)  # v3
            points.append(Point(lon, lat))
    else:
        spacing = float(demand_cfg["grid_spacing_m"])
        area_proj = area.to_crs(proj_crs)
        union_proj = (
            area_proj.union_all() if hasattr(area_proj, "union_all") else area_proj.unary_union
        )
        minx, miny, maxx, maxy = union_proj.bounds
        y = miny
        while y <= maxy:
            x = minx
            while x <= maxx:
                pt = Point(x, y)
                if union_proj.contains(pt):
                    points.append(pt)
                x += spacing
            y += spacing
        gdf = gpd.GeoDataFrame({"demand_id": range(len(points))}, geometry=points, crs=proj_crs)
        return gdf.to_crs(geo_crs)

    gdf = gpd.GeoDataFrame({"demand_id": range(len(points))}, geometry=points, crs=geo_crs)
    return gdf


def assign_access_to_demand(demand, graph, source_nodes, proj_crs: str):
    """Assign each demand point the travel time (minutes) to its nearest facility.

    Snaps demand points to graph nodes, runs the multi-source search, and writes
    a ``travel_time_min`` column. Returns the demand GeoDataFrame.
    """
    import osmnx as ox

    demand_proj = demand.to_crs(proj_crs)
    node_ids = ox.distance.nearest_nodes(
        graph, X=demand_proj.geometry.x.to_numpy(), Y=demand_proj.geometry.y.to_numpy()
    )
    times = compute_access_times(graph, list(source_nodes))
    out = demand.copy()
    out["node_id"] = node_ids
    out["travel_time_min"] = nearest_times_to_minutes(node_ids, times)
    return out


def main() -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    import yaml

    from access.facilities import (
        facility_source_nodes,
        load_facilities,
        snap_facilities_to_nodes,
    )
    from access.network import build_drive_graph

    parser = argparse.ArgumentParser(description="Compute nearest-facility access times.")
    parser.add_argument("osm_path", type=Path)
    parser.add_argument("facilities_path", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "access.gpkg")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    proj_crs = cfg["study_area"]["projected_crs"]

    graph = build_drive_graph(args.osm_path, args.config)
    facilities = snap_facilities_to_nodes(load_facilities(args.facilities_path, proj_crs), graph)
    sources = facility_source_nodes(facilities)

    # Demand surface needs the study-area polygon; here we use the facility hull
    # as a stand-in if no admin boundary is wired in at the CLI level.
    import geopandas as gpd

    study_area = gpd.GeoDataFrame(geometry=[facilities.union_all().convex_hull], crs=proj_crs)
    demand = build_demand_surface(study_area, cfg)
    demand = assign_access_to_demand(demand, graph, sources, proj_crs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    demand.to_file(args.out, driver="GPKG")
    print(f"Wrote {len(demand)} demand points to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
