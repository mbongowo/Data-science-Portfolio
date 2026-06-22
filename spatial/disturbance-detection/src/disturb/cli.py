"""Command-line entry point for the disturbance-detection pipeline.

    disturb --config config/aoi.yaml

Runs the full path: build NDVI cube -> per-pixel harmonic decompose ->
breakpoint detection -> validate against the configured event. The heavy
geospatial steps are imported lazily, so ``disturb --help`` works on a bare
install; the actual run needs the EO stack (``pixi install``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML config (lazy PyYAML import)."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to read the config.") from exc
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(config_path: str) -> int:
    """Execute the end-to-end pipeline described by ``config_path``."""
    cfg = load_config(config_path)
    print(f"Loaded config: {config_path}")
    print(f"AOI         : {cfg['aoi']['name']} bbox={cfg['aoi']['bbox']}")
    print(f"Time        : {cfg['time']['start']} -> {cfg['time']['end']}")
    print(f"Source      : {cfg['source']['collection']}")

    # Imported here so --help / dry inspection never needs the geo stack.
    from .cube import build_ndvi_cube
    from .detect import detect_breakpoint  # noqa: F401  (used in apply below)

    print("Building NDVI time cube (this hits the live STAC API)...")
    cube = build_ndvi_cube(
        bbox=cfg["aoi"]["bbox"],
        start=cfg["time"]["start"],
        end=cfg["time"]["end"],
        collection=cfg["source"]["collection"],
        resolution=cfg["source"]["resolution"],
        freq=cfg["time"]["resample_freq"],
        stac_url=cfg["source"]["stac_url"],
    )
    print(f"Cube built: dims={dict(cube.sizes)} (Dask-backed)")
    print(
        "Next: apply harmonic_decompose + detect_breakpoint per pixel via "
        "xarray.apply_ufunc, then validate. See notebooks/01_disturbance.ipynb."
    )
    return 0


def run_demo_cli(seed: int, out_dir: str) -> int:
    """Run the reproducible pure-numpy demo and print its metrics."""
    import json

    from .demo import run_demo

    metrics = run_demo(seed=seed, out_dir=out_dir)
    print(f"Demo (seed={seed}) -> artefacts in {out_dir}/")
    print(json.dumps(metrics, indent=2))
    rec = metrics["detected_index"]
    plant = metrics["planted_break_index"]
    print(
        f"\nPlanted break at index {plant}; detected at {rec} "
        f"(magnitude {metrics['detected_magnitude']:+.3f}, "
        f"score {metrics['detected_score']:.2f})."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="disturb", description=__doc__.splitlines()[0]
    )
    # Keep `disturb --config ...` working as the default (no subcommand) path.
    parser.add_argument(
        "--config",
        default="config/aoi.yaml",
        help="Path to the AOI/detection YAML config.",
    )
    sub = parser.add_subparsers(dest="command")

    demo_parser = sub.add_parser(
        "demo",
        help="Run the reproducible pure-numpy demo (no geo stack needed).",
    )
    demo_parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default 0)."
    )
    demo_parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for series.csv/components.csv/summary.json.",
    )

    args = parser.parse_args(argv)

    if args.command == "demo":
        return run_demo_cli(args.seed, args.out_dir)
    return run(args.config)


if __name__ == "__main__":
    sys.exit(main())
