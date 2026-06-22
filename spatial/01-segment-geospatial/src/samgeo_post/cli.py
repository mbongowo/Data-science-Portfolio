"""Command-line entry point for samgeo_post.

Three subcommands:

* ``demo``      — dependency-free synthetic run; needs only numpy. Writes
                  ``outputs/region_props.csv`` + ``outputs/summary.json``.
* ``segment``   — pull basemap tiles for the AOI and run SAM (heavy; GPU).
* ``vectorize`` — polygonise a labelled mask GeoTIFF to GeoJSON (heavy).

Only ``demo`` runs without the geospatial / deep-learning stack. The heavy
imports for ``segment`` and ``vectorize`` happen inside their handlers (via the
lazy wrapper modules), so importing this module — e.g. for ``--help`` or in a
test — never requires those dependencies.
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


def _run_segment(args: argparse.Namespace) -> int:
    from samgeo_post.segment import segment_basemap

    cfg = _load_yaml(args.config)
    aoi = cfg["aoi"]["bbox"]
    bbox = (aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"])
    out_tif = segment_basemap(
        aoi_bbox=bbox,
        zoom=args.zoom or cfg["basemap"]["zoom"],
        out_tif=args.out,
        source=cfg["basemap"]["source"],
    )
    print(f"Wrote mask GeoTIFF: {out_tif}")
    return 0


def _run_vectorize(args: argparse.Namespace) -> int:
    import rasterio

    from samgeo_post.vectorize import masks_to_geojson

    cfg = _load_yaml(args.config)
    with rasterio.open(args.mask) as ds:
        labeled = ds.read(1)
        transform = ds.transform
        crs = ds.crs
    gdf = masks_to_geojson(
        labeled,
        transform=transform,
        crs=crs,
        pixel_size_m=cfg["analysis"]["pixel_size_m"],
        out_path=args.out,
    )
    print(f"Wrote {len(gdf)} polygons to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="samgeo-post", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo", help="Dependency-free synthetic analytics run (numpy only)."
    )
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out", default="outputs")

    p_seg = sub.add_parser(
        "segment", help="Pull basemap tiles and run SAM (heavy; needs a GPU)."
    )
    p_seg.add_argument("--config", default="config/aoi.yaml")
    p_seg.add_argument("--zoom", type=int, default=None)
    p_seg.add_argument("--out", default="outputs/douala_masks.tif")

    p_vec = sub.add_parser(
        "vectorize", help="Polygonise a labelled mask GeoTIFF to GeoJSON (heavy)."
    )
    p_vec.add_argument("--config", default="config/aoi.yaml")
    p_vec.add_argument(
        "--mask", required=True, help="Path to the labelled mask GeoTIFF."
    )
    p_vec.add_argument("--out", default="outputs/features.geojson")

    args = parser.parse_args(argv)

    if args.command == "demo":
        from samgeo_post.demo import run_demo

        summary = run_demo(seed=args.seed, out_dir=args.out)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "segment":
        return _run_segment(args)

    if args.command == "vectorize":
        return _run_vectorize(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
