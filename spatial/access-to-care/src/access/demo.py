"""Runnable, self-contained demo of the access-to-care pipeline.

Synthesizes a small seeded road network (a regular grid graph with
``travel_time`` edge weights), places a handful of facilities, draws a
population per node, and runs the real routing and equity functions end to end
on it -- no OSM, no rasters, no geospatial stack. The network is synthetic and
seeded, so the numbers are reproducible but illustrative, not measured from
real Cameroon data.

Run it with ``python -m access.demo`` (or ``pixi run demo`` / ``make demo``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from access.access import assign_nearest_source, nearest_times_to_minutes
from access.equity import coverage_bands, population_within_thresholds
from access.metrics import facility_load, gini_coefficient

THRESHOLDS_MIN = [30, 60, 120]


def _grid_adjacency(
    n: int, rng: np.random.Generator, drop_prob: float = 0.18
) -> dict[int, list[tuple[int, float]]]:
    """Build an ``n x n`` 4-connected grid graph with random travel-time weights.

    Node id is ``row * n + col``. Each edge gets a travel time in seconds drawn
    from a uniform range (roughly 6--20 minutes per grid hop), assigned
    symmetrically so the graph is undirected. A fraction ``drop_prob`` of edges
    are removed to model a sparser rural network, which lengthens some routes and
    can strand a few nodes (so the thresholds and the unreachable bucket are not
    all trivially zero).
    """
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n * n)}
    for r in range(n):
        for c in range(n):
            node = r * n + c
            if c + 1 < n and rng.random() > drop_prob:  # edge to the right
                w = float(rng.uniform(360.0, 1200.0))
                right = node + 1
                adjacency[node].append((right, w))
                adjacency[right].append((node, w))
            if r + 1 < n and rng.random() > drop_prob:  # edge to the neighbour below
                w = float(rng.uniform(360.0, 1200.0))
                down = node + n
                adjacency[node].append((down, w))
                adjacency[down].append((node, w))
    return adjacency


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the full demo and write artifacts; return headline metrics.

    Builds a seeded grid network, picks facility nodes, draws a population per
    node, routes every node to its nearest facility with the real multi-source
    Dijkstra, and computes population-weighted coverage. Writes ``demand.csv``
    and ``summary.json`` to ``out_dir``.

    Returns a dict with ``n_nodes``, ``n_facilities``, ``population_total``,
    ``share_within_30min``, ``share_within_60min`` and ``pop_unreachable``.
    """
    rng = np.random.default_rng(seed)
    grid_n = 12
    n_nodes = grid_n * grid_n
    adjacency = _grid_adjacency(grid_n, rng)

    # A handful of facilities spread across the grid (two corners + centre).
    facilities = [0, n_nodes - 1, (grid_n // 2) * grid_n + grid_n // 2]

    # Population per node: a positive, lumpy surface (gamma-distributed).
    population = rng.gamma(shape=2.0, scale=400.0, size=n_nodes).round(1)

    # Route every node to its nearest facility (seconds) and which one.
    times_seconds, nearest = assign_nearest_source(adjacency, facilities)
    node_ids = list(range(n_nodes))
    travel_time_min = nearest_times_to_minutes(node_ids, times_seconds)

    # Tidy demand frame. Admin2 = grid quadrant, so equity has >1 unit to compare.
    half = grid_n // 2
    admin2 = []
    for node in node_ids:
        r, c = divmod(node, grid_n)
        ns = "N" if r < half else "S"
        ew = "W" if c < half else "E"
        admin2.append(f"{ns}{ew}")

    demand = pd.DataFrame(
        {
            "node_id": node_ids,
            "admin2": admin2,
            "travel_time_min": travel_time_min,
            "population": population,
            "nearest_facility": [nearest.get(n) for n in node_ids],
        }
    )

    within = population_within_thresholds(
        demand["travel_time_min"], demand["population"], THRESHOLDS_MIN
    )
    bands = coverage_bands(demand["travel_time_min"], demand["population"], THRESHOLDS_MIN)
    gini_tt = gini_coefficient(demand["travel_time_min"], demand["population"])
    load = facility_load(demand["node_id"], demand["nearest_facility"], demand["population"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demand.to_csv(out_dir / "demand.csv", index=False)

    summary = {
        "seed": seed,
        "n_nodes": n_nodes,
        "n_facilities": len(facilities),
        "thresholds_min": THRESHOLDS_MIN,
        "population_within_thresholds": within,
        "coverage_bands": bands,
        "gini_travel_time": gini_tt,
        "facility_load": {str(k): v for k, v in load.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "n_nodes": n_nodes,
        "n_facilities": len(facilities),
        "population_total": within["population_total"],
        "share_within_30min": within["share_within_30min"],
        "share_within_60min": within["share_within_60min"],
        "pop_unreachable": bands["pop_unreachable"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the seeded access-to-care demo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default="outputs")
    args = parser.parse_args()

    metrics = run_demo(seed=args.seed, out_dir=args.out)
    print("access-to-care demo (seeded synthetic network)")
    print(f"  nodes={metrics['n_nodes']}  facilities={metrics['n_facilities']}")
    print(f"  population_total = {metrics['population_total']:.1f}")
    print(f"  share within 30 min = {metrics['share_within_30min']:.1%}")
    print(f"  share within 60 min = {metrics['share_within_60min']:.1%}")
    print(f"  population unreachable = {metrics['pop_unreachable']:.1f}")
    print(f"  artifacts -> {args.out}/demand.csv, {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
