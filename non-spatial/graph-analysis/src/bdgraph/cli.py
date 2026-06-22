"""Command-line entry point for bdgraph.

Three subcommands map to the three reported algorithms::

    bdgraph pagerank    --config config/graph.yaml --out outputs
    bdgraph communities --config config/graph.yaml --out outputs
    bdgraph triangles   --config config/graph.yaml --out outputs

Each loads the edge list named in the config, runs the algorithm, and writes a
JSON summary (plus a CSV of per-node results) to the output directory.

A fourth subcommand needs no config or data::

    bdgraph demo --seed 0 --out outputs

It synthesises a small seeded stochastic block model and runs the whole
pure-numpy core on it end-to-end (see :mod:`bdgraph.demo`).

The numeric core (`bdgraph.pagerank`, `.components`, `.community`, `.triangles`)
is pure numpy and dense, so it is meant for graphs up to a few thousand nodes.
For a SNAP-scale graph use the Spark GraphFrames pipeline
(`bdgraph.graphframes_pipeline`); the heavy imports there are lazy. This CLI
imports numpy/pandas/yaml lazily inside the handlers so ``bdgraph --help``
stays cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _read_edges(cfg: dict[str, Any]) -> tuple[int, NDArray[Any], dict[int, int]]:
    """Read the edge list into a 0-based contiguous index space.

    Returns ``(n, edges, id_map)`` where ``edges`` is an ``(m, 2)`` int array of
    re-indexed endpoints and ``id_map`` maps the new index back to the original
    node id.
    """
    import numpy as np

    gcfg = cfg["graph"]
    comment = gcfg.get("comment", "#")
    delimiter = gcfg.get("delimiter")

    src: list[int] = []
    dst: list[int] = []
    # utf-8-sig tolerates a leading BOM if the edge file happens to carry one.
    with open(gcfg["edge_file"], encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) < 2:
                continue
            src.append(int(parts[0]))
            dst.append(int(parts[1]))

    ids = sorted(set(src) | set(dst))
    index = {node: i for i, node in enumerate(ids)}
    id_map = {i: node for node, i in index.items()}
    edges = np.array(
        [[index[s], index[d]] for s, d in zip(src, dst, strict=True)], dtype=int
    ).reshape(-1, 2)
    return len(ids), edges, id_map


def _dense_adjacency(n: int, edges: NDArray[Any], *, directed: bool) -> Any:
    import numpy as np

    a = np.zeros((n, n), dtype=float)
    for s, d in edges:
        a[s, d] = 1.0
        if not directed:
            a[d, s] = 1.0
    return a


def _cmd_pagerank(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    import numpy as np

    from bdgraph.pagerank import pagerank

    n, edges, id_map = _read_edges(cfg)
    a = _dense_adjacency(n, edges, directed=cfg["graph"].get("directed", True))
    pcfg = cfg["pagerank"]
    pr = pagerank(
        a,
        damping=pcfg.get("damping", 0.85),
        max_iter=pcfg.get("max_iter", 100),
        tol=float(pcfg.get("tol", 1e-9)),
    )
    top_k = int(pcfg.get("top_k", 20))
    order = np.argsort(pr)[::-1][:top_k]
    top = [{"node": int(id_map[int(i)]), "pagerank": float(pr[i])} for i in order]
    summary = {"algorithm": "pagerank", "n_nodes": n, "top": top}
    _write(out, "pagerank", summary)
    return summary


def _cmd_communities(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    import numpy as np

    from bdgraph.community import label_propagation
    from bdgraph.components import num_components

    n, edges, id_map = _read_edges(cfg)
    a = _dense_adjacency(n, edges, directed=False)
    ccfg = cfg["communities"]
    labels = label_propagation(
        a, max_iter=ccfg.get("max_iter", 10), seed=ccfg.get("seed", 0)
    )
    n_comm = int(len(np.unique(labels)))
    n_cc = num_components(n, [(int(s), int(d)) for s, d in edges])
    summary = {
        "algorithm": "label_propagation",
        "n_nodes": n,
        "n_communities": n_comm,
        "n_connected_components": n_cc,
    }
    _write(out, "communities", summary)
    return summary


def _cmd_triangles(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    from bdgraph.triangles import triangle_count

    n, edges, id_map = _read_edges(cfg)
    a = _dense_adjacency(n, edges, directed=False)
    total = triangle_count(a)
    summary = {"algorithm": "triangle_count", "n_nodes": n, "triangles": total}
    _write(out, "triangles", summary)
    return summary


def _cmd_demo(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    from bdgraph.demo import run_demo

    return run_demo(seed=int(cfg.get("seed", 0)), out_dir=out)


def _write(out: Path, name: str, summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bdgraph", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pagerank", "communities", "triangles"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="config/graph.yaml")
        p.add_argument("--out", default="outputs")
    # `demo` needs no config or downloaded data: it synthesises its own graph.
    pd = sub.add_parser("demo")
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--out", default="outputs")

    args = parser.parse_args(argv)
    out = Path(args.out)

    if args.command == "demo":
        cfg: dict[str, Any] = {"seed": args.seed}
    else:
        cfg = _load_config(args.config)

    handlers = {
        "pagerank": _cmd_pagerank,
        "communities": _cmd_communities,
        "triangles": _cmd_triangles,
        "demo": _cmd_demo,
    }
    summary = handlers[args.command](cfg, out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
