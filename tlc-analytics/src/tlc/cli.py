"""Command-line entry point for tlc-analytics.

Two subcommands:

* ``tlc mart``      — build the aggregated marts and write them to ``--out``.
* ``tlc benchmark`` — run the engine bake-off and write a ranking summary.

Heavy work (reading Parquet, the engine runners) is imported lazily inside the
command bodies, so importing this module — for ``--help`` or testing — never
requires pandas-heavy I/O or the engine stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_app() -> Any:
    import typer  # lazy: CLI-only dependency

    app = typer.Typer(add_completion=False, help=__doc__)

    @app.command()
    def mart(
        config: str = "config/tlc.yaml",
        out: str = "outputs",
    ) -> None:
        """Build the aggregated marts from the pandas reference path."""
        import pandas as pd

        from tlc import marts
        from tlc.clean import clean_trips

        cfg = _load_config(config)
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)

        glob = cfg["data"]["parquet_glob"]
        fare_cap = cfg.get("clean", {}).get("fare_cap", 500.0)

        raw = pd.read_parquet(glob)
        trips = clean_trips(raw, fare_cap=fare_cap)

        builders = {
            "hourly_demand": marts.hourly_demand,
            "demand_by_dow": marts.demand_by_dow,
            "tip_rate_by_payment": marts.tip_rate_by_payment,
            "fare_summary": marts.fare_summary,
        }
        for name, fn in builders.items():
            fn(trips).to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"Wrote {len(builders)} marts to {out_dir}")

    @app.command()
    def benchmark(
        config: str = "config/tlc.yaml",
        out: str = "outputs",
    ) -> None:
        """Run the engine bake-off and write a ranking summary."""
        from tlc.benchmark import summarize

        cfg = _load_config(config)
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)

        from tlc import engines  # noqa: F401  (lazy; engine stack required)

        queries = [q["name"] for q in cfg.get("queries", [])]
        engine_names = cfg.get("engines", [])
        print(f"Configured: {len(engine_names)} engines x {len(queries)} queries")

        # Engine execution is environment-specific (cluster / warehouse creds);
        # wire run_duckdb / run_spark to the configured queries here, collect
        # BenchmarkResult rows, then rank and persist them.
        results: list[Any] = []
        ranking = summarize(results) if results else None
        if ranking is not None:
            ranking.to_csv(out_dir / "benchmark.csv", index=False)
            print(json.dumps(ranking.to_dict(orient="records"), indent=2))
        else:
            print("No engine results collected; see USAGE.md to wire the runners.")

    return app


def main(argv: list[str] | None = None) -> int:
    app = _build_app()
    app(args=argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
