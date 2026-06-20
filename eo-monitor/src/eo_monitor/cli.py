"""Typer CLI: ``eo-monitor run --config config/corn_belt.yaml``.

Orchestrates search -> cube -> indices -> anomaly -> COG export with logging.
Heavy geospatial imports are deferred into ``run`` so ``--help`` and config
validation stay fast and importable without the full geo stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from eo_monitor import __version__
from eo_monitor.config import load_config

app = typer.Typer(
    add_completion=False,
    help="Sentinel-2 vegetation/moisture anomaly monitoring (STAC -> COG).",
)

logger = logging.getLogger("eo_monitor")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"eo-monitor {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """eo-monitor command-line interface."""


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=False,
        help="Path to the YAML configuration file.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override the output directory from the config.",
    ),
    max_items: Optional[int] = typer.Option(
        None,
        "--max-items",
        help="Override the hard cap on STAC items pulled.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
) -> None:
    """Run the full anomaly pipeline from a config file."""
    _configure_logging(verbose)
    cfg = load_config(config)
    if output_dir is not None:
        cfg.output.dir = output_dir
    if max_items is not None:
        cfg.max_items = max_items

    logger.info("Loaded config: %s", config)
    logger.info("AOI=%s  target=%s  baseline=%s", cfg.aoi, cfg.date_range, cfg.baseline)

    # Deferred heavy imports (keep --help / config-only usage light).
    from eo_monitor import anomaly as anom
    from eo_monitor import io as eo_io
    from eo_monitor.catalog import search_for_config
    from eo_monitor.cube import load_cube
    from eo_monitor.indices import compute_index

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover items for the target and baseline windows.
    logger.info("[1/5] Searching STAC for target window...")
    target_items = search_for_config(cfg)
    logger.info("[1/5] Searching STAC for baseline window...")
    baseline_items = search_for_config(cfg, window=cfg.baseline)

    # 2. Build lazy, masked cubes.
    logger.info("[2/5] Building masked cubes (lazy / Dask)...")
    target_cube = load_cube(target_items, cfg)
    baseline_cube = load_cube(baseline_items, cfg)

    # Optionally restrict the baseline to its climatological months.
    if cfg.baseline.months and "time" in baseline_cube.dims:
        month_sel = baseline_cube["time"].dt.month.isin(cfg.baseline.months)
        baseline_cube = baseline_cube.sel(time=month_sel)

    for index_name in cfg.indices:
        logger.info("[3/5] Computing %s (target + baseline)...", index_name)
        target_bands = {b: target_cube[b] for b in target_cube.data_vars if b != "SCL"}
        baseline_bands = {b: baseline_cube[b] for b in baseline_cube.data_vars if b != "SCL"}

        target_idx = compute_index(index_name, target_bands)
        baseline_idx = compute_index(index_name, baseline_bands)

        # Median composite of the target window -> single map per index.
        target_composite = (
            target_idx.median(dim="time", skipna=True)
            if "time" in getattr(target_idx, "dims", ())
            else target_idx
        )

        logger.info("[4/5] Computing %s z-score anomaly vs baseline...", index_name)
        z = anom.anomaly_cube(target_composite, baseline_idx, dim="time")

        # 5. Export COG + quicklook.
        logger.info("[5/5] Exporting %s anomaly outputs...", index_name)
        stem = index_name.lower()
        cog_path = out_dir / f"{stem}_anomaly.tif"
        eo_io.write_cog(z, cog_path)
        # Also export the raw target composite for reference.
        eo_io.write_cog(target_composite, out_dir / f"{stem}_composite.tif")

        if cfg.output.write_quicklook:
            eo_io.write_quicklook(
                z,
                out_dir / f"{stem}_anomaly.png",
                cmap="RdYlGn",
                vmin=-3,
                vmax=3,
                title=f"{index_name} z-score anomaly",
            )

    logger.info("Done. Outputs written to %s", out_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    app()
