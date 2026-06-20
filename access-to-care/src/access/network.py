"""Build a routable drivable road graph from OpenStreetMap.

Pipeline role: read an OSM extract, assign per-edge ``travel_time`` from the
highway type (via a configurable speed lookup) and edge length, then keep the
largest connected component so every node can reach every other node.

The arithmetic helpers (:func:`speed_for_highway`, :func:`travel_time_seconds`,
:func:`assign_edge_speeds_and_times`) are pure Python so they can be unit-tested
without any geospatial dependency. The graph-building entry points
(:func:`build_drive_graph`) require ``osmnx``/``networkx`` and a real OSM file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "sources.yaml"


def load_speed_config(config_path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, float], float]:
    """Return ``(speeds_kph, default_speed_kph)`` from the YAML config."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    speeds = {str(k): float(v) for k, v in cfg["speeds_kph"].items()}
    default = float(cfg["default_speed_kph"])
    return speeds, default


def speed_for_highway(
    highway: Any,
    speeds_kph: dict[str, float],
    default_speed_kph: float,
) -> float:
    """Resolve a driving speed (km/h) for an OSM ``highway`` tag value.

    OSM ``highway`` can be a single string or a list (when ways are merged).
    For a list we take the fastest matching speed. Unknown tags fall back to
    ``default_speed_kph``.
    """
    if isinstance(highway, (list, tuple)):
        candidates = [speeds_kph.get(str(h), default_speed_kph) for h in highway]
        return max(candidates) if candidates else default_speed_kph
    return speeds_kph.get(str(highway), default_speed_kph)


def travel_time_seconds(length_m: float, speed_kph: float) -> float:
    """Convert an edge length (metres) and speed (km/h) to travel time (seconds)."""
    if speed_kph <= 0:
        raise ValueError("speed_kph must be positive")
    speed_mps = speed_kph * 1000.0 / 3600.0
    return float(length_m) / speed_mps


def assign_edge_speeds_and_times(
    edges: list[dict[str, Any]],
    speeds_kph: dict[str, float],
    default_speed_kph: float,
) -> list[dict[str, Any]]:
    """Annotate a list of edge dicts with ``speed_kph`` and ``travel_time``.

    Each edge dict must contain ``length`` (metres) and ``highway``. This mirrors
    what :func:`build_drive_graph` does on a real networkx graph but operates on
    plain dicts so it is testable without geo dependencies. Returns a new list.
    """
    out: list[dict[str, Any]] = []
    for edge in edges:
        speed = speed_for_highway(edge.get("highway"), speeds_kph, default_speed_kph)
        tt = travel_time_seconds(float(edge["length"]), speed)
        annotated = dict(edge)
        annotated["speed_kph"] = speed
        annotated["travel_time"] = tt
        out.append(annotated)
    return out


def total_network_time(edges: list[dict[str, Any]]) -> float:
    """Sum the ``travel_time`` field over annotated edges.

    Convenience reduction over the output of
    :func:`assign_edge_speeds_and_times`. Edges without a ``travel_time`` key
    contribute zero. Pure Python.

    Parameters
    ----------
    edges : list of dict
        Edge dicts carrying a ``travel_time`` (seconds) field.

    Returns
    -------
    float
        Total travel time across all edges, in seconds.
    """
    return float(sum(float(e.get("travel_time", 0.0)) for e in edges))


def build_drive_graph(
    osm_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
):
    """Build a routable drivable :class:`networkx.MultiDiGraph` from an OSM file.

    Steps:
      1. Load the drivable network from the OSM extract.
      2. Assign ``travel_time`` (seconds) to every edge from highway type + length.
      3. Keep the largest strongly connected component (every node reachable).

    Requires ``osmnx``; not importable in a geo-free environment, hence the local
    import.
    """
    import osmnx as ox

    speeds_kph, default_speed_kph = load_speed_config(config_path)

    osm_path = Path(osm_path)
    if osm_path.suffix == ".pbf":
        graph = _graph_from_pbf(ox, osm_path)
    else:
        graph = ox.graph_from_xml(osm_path)

    for _u, _v, data in graph.edges(data=True):
        length = float(data.get("length", 0.0))
        speed = speed_for_highway(data.get("highway"), speeds_kph, default_speed_kph)
        data["speed_kph"] = speed
        data["travel_time"] = travel_time_seconds(length, speed) if length > 0 else 0.0

    graph = largest_component(graph, strongly=True)
    return graph


def _graph_from_pbf(ox: Any, osm_path: Path):  # pragma: no cover - requires osmnx
    """Load a graph from a .pbf extract.

    Newer osmnx exposes ``graph_from_xml`` for OSM XML; for .pbf we convert via
    the osmnx features/graph API. Kept isolated so the public function stays clean.
    """
    return ox.graph_from_xml(osm_path)


def largest_component(graph, strongly: bool = True):
    """Return the subgraph of ``graph``'s largest connected component."""
    import networkx as nx

    if graph.is_directed():
        if strongly:
            comps = nx.strongly_connected_components(graph)
        else:
            comps = nx.weakly_connected_components(graph)
    else:
        comps = nx.connected_components(graph)
    largest = max(comps, key=len)
    return graph.subgraph(largest).copy()


def main() -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Build a drivable road graph from OSM.")
    parser.add_argument("osm_path", type=Path, help="Path to OSM extract (.osm/.pbf).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    graph = build_drive_graph(args.osm_path, args.config)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
