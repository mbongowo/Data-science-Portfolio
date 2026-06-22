"""Command-line entry point for the dbt-modern-stack pipeline.

Three subcommands wrap the stack:

    dwh seed  --config config/warehouse.yaml    # raw IMDb extracts -> DuckDB
    dwh build --config config/warehouse.yaml    # dbt build (models + tests)
    dwh dq    --config config/warehouse.yaml    # the pure-pandas DQ core

The heavy imports (typer, duckdb, dbt) happen inside the command functions, so
importing this module never requires the full stack. ``main`` falls back to a
small stdlib ``argparse`` parser if typer is not installed, so the entry point
stays usable in a minimal environment.
"""

from __future__ import annotations

import sys

DEFAULT_CONFIG = "config/warehouse.yaml"


def _cmd_seed(config: str) -> int:
    from dwh.orchestration import seed_warehouse

    loaded = seed_warehouse(config)
    for name, rows in loaded.items():
        print(f"loaded raw.{name}: {rows} rows")
    return 0


def _cmd_build(config: str) -> int:
    from dwh.orchestration import run_dbt_build

    return run_dbt_build()


def _cmd_dq(config: str) -> int:
    """Run the pure-pandas data-quality core against the loaded warehouse.

    Reads each configured source table out of DuckDB and runs the generic-test
    suite defined for the marts, printing a ``dbt test``-style summary. Returns
    a non-zero exit code if any test fails.
    """
    import duckdb

    from dwh.dq import TestSpec, run_suite
    from dwh.orchestration import load_config

    cfg = load_config(config)
    db_path = cfg["warehouse"]["duckdb_path"]
    raw_schema = cfg["warehouse"].get("raw_schema", "raw")

    con = duckdb.connect(db_path, read_only=True)
    try:
        titles = con.execute(f"SELECT * FROM {raw_schema}.title_basics").df()
        ratings = con.execute(f"SELECT * FROM {raw_schema}.title_ratings").df()
    finally:
        con.close()

    suite = [
        TestSpec("not_null", titles, "tconst", table="stg_titles"),
        TestSpec("unique", titles, "tconst", table="stg_titles"),
        TestSpec("not_null", ratings, "tconst", table="stg_ratings"),
        TestSpec(
            "relationships",
            ratings,
            "tconst",
            table="fct_title_rating",
            parent=titles,
            parent_col="tconst",
        ),
    ]
    summary = run_suite(suite)
    print(summary.to_string(index=False))
    return 0 if bool(summary["passed"].all()) else 1


def _build_typer_app():  # type: ignore[no-untyped-def]
    import typer

    app = typer.Typer(add_completion=False, help=__doc__)

    @app.command()
    def seed(config: str = DEFAULT_CONFIG) -> None:
        """Load raw IMDb extracts into the DuckDB warehouse."""
        raise typer.Exit(_cmd_seed(config))

    @app.command()
    def build(config: str = DEFAULT_CONFIG) -> None:
        """Run `dbt build` (all models + generic/singular tests)."""
        raise typer.Exit(_cmd_build(config))

    @app.command()
    def dq(config: str = DEFAULT_CONFIG) -> None:
        """Run the pure-pandas data-quality core."""
        raise typer.Exit(_cmd_dq(config))

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

    parser = argparse.ArgumentParser(prog="dwh", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("seed", "build", "dq"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    dispatch = {"seed": _cmd_seed, "build": _cmd_build, "dq": _cmd_dq}
    return dispatch[args.command](args.config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
