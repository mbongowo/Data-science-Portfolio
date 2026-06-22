"""Command-line entry point for floodmap.

Two subcommands:

* ``demo`` — dependency-free synthetic run; needs only numpy. Writes
             ``outputs/flood_stats.json`` + ``outputs/flood_mask.npy``.
* ``map``  — pull Sentinel-1 GRD pre/post backscatter from STAC for the AOI in
             ``config/aoi.yaml``, Otsu-threshold each, derive water masks,
             compute the before/after flood extent, and print the hectares
             (heavy; needs the geospatial stack).

Only ``demo`` runs without the geospatial stack. The heavy imports for ``map``
happen inside its handler (via the lazy ``floodmap.stac`` module), so importing
this module — for ``--help`` or in a test — never requires those dependencies.
The default real-data path is the auth-free Earth Search STAC catalogue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_map(args: argparse.Namespace) -> int:
    from floodmap.change import flood_extent, flood_stats
    from floodmap.stac import aoi_to_bbox, build_flood_inputs
    from floodmap.threshold import otsu_threshold
    from floodmap.water import water_mask

    cfg = _load_yaml(args.config)
    bbox = aoi_to_bbox(cfg)
    a = cfg["analysis"]

    pre_db, post_db = build_flood_inputs(
        bbox,
        pre_date=cfg["pre_date"],
        post_date=cfg["post_date"],
        orbit=a.get("orbit", "descending"),
        polarization=a.get("polarization", "vh"),
        resolution=a["pixel_size_m"],
    )
    pre_water = water_mask(pre_db, otsu_threshold(pre_db), polarity="below")
    post_water = water_mask(post_db, otsu_threshold(post_db), polarity="below")
    masks = flood_extent(pre_water, post_water)
    stats = flood_stats(masks, a["pixel_size_m"])
    print(json.dumps(stats, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="floodmap", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo", help="Dependency-free synthetic SAR flood-mapping run (numpy)."
    )
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out", default="outputs")

    p_map = sub.add_parser(
        "map",
        help="Sentinel-1 before/after flood extent in hectares from STAC (heavy).",
    )
    p_map.add_argument("--config", default="config/aoi.yaml")

    args = parser.parse_args(argv)

    if args.command == "demo":
        from floodmap.demo import run_demo

        summary = run_demo(seed=args.seed, out_dir=args.out)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "map":
        return _run_map(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
