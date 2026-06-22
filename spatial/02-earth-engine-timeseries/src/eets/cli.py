"""Command-line entry point for eets (earth-engine time series).

Three subcommands:

* ``demo``       — dependency-free synthetic run; needs only numpy. Writes
                   ``outputs/index_timeseries.csv`` + ``outputs/change_stats.json``.
* ``timeseries`` — load a Sentinel-2 index composite for one period from STAC and
                   print the AOI mean (heavy; needs the geo stack).
* ``change``     — load baseline + recent composites from STAC, classify the
                   change, and print loss / gain hectares (heavy).

Only ``demo`` runs without the geospatial stack. The heavy imports for
``timeseries`` and ``change`` happen inside their handlers (via the lazy
``eets.stac`` module), so importing this module — for ``--help`` or in a test —
never requires those dependencies. The default real-data path is the auth-free
Earth Search STAC catalogue; the Earth Engine route lives in ``eets.gee``.
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


def _bbox_and_cfg(config: str | Path) -> tuple[list[float], dict[str, Any]]:
    cfg = _load_yaml(config)
    b = cfg["aoi"]["bbox"]
    bbox = [b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"]]
    return bbox, cfg


def _run_timeseries(args: argparse.Namespace) -> int:
    import numpy as np

    from eets.stac import load_s2_period

    bbox, cfg = _bbox_and_cfg(args.config)
    composite = load_s2_period(
        bbox,
        args.start,
        args.end,
        max_cloud=cfg["analysis"]["max_cloud"],
        index=args.index or cfg["analysis"]["index"],
        resolution=cfg["analysis"]["pixel_size_m"],
    )
    mean = float(np.nanmean(composite))
    print(
        f"AOI mean {args.index or cfg['analysis']['index']} "
        f"{args.start}..{args.end}: {mean:.4f}"
    )
    return 0


def _run_change(args: argparse.Namespace) -> int:
    from eets.change import change_map, change_stats, classify_change
    from eets.stac import build_change_inputs

    bbox, cfg = _bbox_and_cfg(args.config)
    a = cfg["analysis"]
    before, after = build_change_inputs(
        bbox,
        baseline_years=(cfg["baseline_years"]["start"], cfg["baseline_years"]["end"]),
        recent_years=(cfg["recent_years"]["start"], cfg["recent_years"]["end"]),
        index=a["index"],
        max_cloud=a["max_cloud"],
        resolution=a["pixel_size_m"],
    )
    delta = change_map(before, after)
    classified = classify_change(delta, a["loss_thresh"], a["gain_thresh"])
    stats = change_stats(classified, a["pixel_size_m"])
    print(json.dumps(stats, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eets", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo", help="Dependency-free synthetic time-series + change run (numpy)."
    )
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out", default="outputs")

    p_ts = sub.add_parser(
        "timeseries", help="Load one S2 index composite from STAC (heavy)."
    )
    p_ts.add_argument("--config", default="config/aoi.yaml")
    p_ts.add_argument("--start", required=True, help="ISO start date.")
    p_ts.add_argument("--end", required=True, help="ISO end date.")
    p_ts.add_argument("--index", default=None, help="ndvi | ndwi | nbr.")

    p_ch = sub.add_parser(
        "change", help="Baseline vs recent change in hectares from STAC (heavy)."
    )
    p_ch.add_argument("--config", default="config/aoi.yaml")

    args = parser.parse_args(argv)

    if args.command == "demo":
        from eets.demo import run_demo

        summary = run_demo(seed=args.seed, out_dir=args.out)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "timeseries":
        return _run_timeseries(args)

    if args.command == "change":
        return _run_change(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
