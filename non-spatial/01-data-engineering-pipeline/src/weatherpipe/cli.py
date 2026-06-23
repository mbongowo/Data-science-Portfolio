"""Command-line entry point for the Cameroon weather pipeline.

Subcommands wrap the stack:

    weatherpipe demo   --out-dir outputs        # end-to-end core on synthetic data
    weatherpipe ingest --config config/config.yaml   # Open-Meteo -> partitioned lake
    weatherpipe build  --config config/config.yaml   # load warehouse + dbt build

Only ``demo`` runs on the dependency-light core. ``ingest`` and ``build`` import
their heavy dependencies (requests, duckdb, dbt) lazily inside the command
functions, so importing this module never requires the full stack. ``main`` uses
typer when available and falls back to a stdlib ``argparse`` parser otherwise, so
the entry point stays usable in a minimal environment.
"""

from __future__ import annotations

import sys

DEFAULT_CONFIG = "config/config.yaml"


def _cmd_demo(out_dir: str = "outputs", seed: int = 0) -> int:
    """Run the pure-pandas core end to end on seeded synthetic data."""
    import json

    from weatherpipe.demo import run_demo

    summary = run_demo(seed=seed, out_dir=out_dir)
    print(json.dumps(summary, indent=2))
    h = summary["hottest_station_month"]
    print(
        f"\n{summary['n_records']} clean records, {summary['n_stations']} stations, "
        f"{summary['n_rejected']} rejected ({summary['pct_valid'] * 100:.2f}% valid). "
        f"Hottest: {h['station']} {h['year']}-{h['month']:02d} "
        f"@ {h['tmean']} C. Artifacts in {out_dir}/."
    )
    return 0


def _cmd_ingest(config: str) -> int:
    """Fetch Open-Meteo history for every configured station into the lake.

    Writes one partitioned Parquet dataset per station under the lake root, using
    the ``station=/year=/month=`` layout. requests / pyarrow are imported lazily.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from weatherpipe.ingest import (
        fetch_open_meteo,
        normalize,
        parse_open_meteo,
        partition_path,
    )
    from weatherpipe.orchestrate import load_config

    cfg = load_config(config)
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    lake_root = cfg.get("lake", {}).get("root", "data/lake")

    from pathlib import Path

    total = 0
    for station in cfg["stations"]:
        payload = fetch_open_meteo(station["lat"], station["lon"], start, end)
        df = normalize(parse_open_meteo(payload), station["name"])
        # Partition by station/year/month using the canonical path logic.
        df = df.assign(
            _part=[
                partition_path(s, d)
                for s, d in zip(df["station"], df["date"], strict=True)
            ]
        )
        for part, group in df.groupby("_part"):
            out_dir = Path(lake_root) / part
            out_dir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(group.drop(columns="_part"))
            pq.write_table(table, out_dir / "data.parquet")
            total += len(group)
        print(f"ingested {station['name']}: {len(df)} rows")
    print(f"total {total} rows under {lake_root}/")
    return 0


def _cmd_build(config: str) -> int:
    """Load the lake into DuckDB and run ``dbt build`` (models + tests)."""
    import glob

    import pyarrow.parquet as pq

    from weatherpipe.orchestrate import load_config
    from weatherpipe.warehouse import load_dataframe

    cfg = load_config(config)
    db_path = cfg["warehouse"]["duckdb_path"]
    lake_root = cfg.get("lake", {}).get("root", "data/lake")

    files = glob.glob(f"{lake_root}/**/data.parquet", recursive=True)
    if files:
        import pandas as pd

        frames = [pq.read_table(f).to_pandas() for f in files]
        combined = pd.concat(frames, ignore_index=True)
        rows = load_dataframe(combined, db_path, "weather", schema="raw")
        print(f"loaded raw.weather: {rows} rows")
    else:
        print(f"no parquet under {lake_root}/; run `weatherpipe ingest` first")

    return run_dbt_build()


def run_dbt_build(
    project_dir: str = "transform",
    profiles_dir: str = "transform",
    *,
    select: str | None = None,
) -> int:
    """Invoke ``dbt build`` via the programmatic API (lazy import). 0 on success."""
    from dbt.cli.main import dbtRunner

    args = [
        "build",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(profiles_dir),
    ]
    if select:
        args += ["--select", select]
    res = dbtRunner().invoke(args)
    return 0 if res.success else 1


def _build_typer_app():  # type: ignore[no-untyped-def]
    import typer

    app = typer.Typer(add_completion=False, help=__doc__)

    @app.command()
    def demo(out_dir: str = "outputs", seed: int = 0) -> None:
        """Run the core end to end on seeded synthetic data (no warehouse)."""
        raise typer.Exit(_cmd_demo(out_dir, seed))

    @app.command()
    def ingest(config: str = DEFAULT_CONFIG) -> None:
        """Fetch Open-Meteo history into the partitioned lake."""
        raise typer.Exit(_cmd_ingest(config))

    @app.command()
    def build(config: str = DEFAULT_CONFIG) -> None:
        """Load the warehouse and run `dbt build` (models + tests)."""
        raise typer.Exit(_cmd_build(config))

    return app


def main(argv: list[str] | None = None) -> int:
    """Entry point. Uses typer if available, else a stdlib argparse fallback."""
    try:
        import typer  # noqa: F401
    except ImportError:
        return _main_argparse(argv)

    app = _build_typer_app()
    app(args=argv, standalone_mode=False)
    return 0


def _main_argparse(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="weatherpipe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--out-dir", default="outputs")
    p_demo.add_argument("--seed", type=int, default=0)

    for name in ("ingest", "build"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_CONFIG)

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args.out_dir, args.seed)
    dispatch = {"ingest": _cmd_ingest, "build": _cmd_build}
    return dispatch[args.command](args.config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
