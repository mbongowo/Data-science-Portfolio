"""Command-line entry point for the crypto-backtest pipeline.

Three subcommands cover the workflow:

    backtest demo   --seed 0 --out outputs
    backtest bars   --config config/strategy.yaml --ticks ... --out ...
    backtest run    --config config/strategy.yaml --bars  ... --out outputs
    backtest report --config config/strategy.yaml --equity ... --out outputs

The ``demo`` subcommand drives the whole pipeline on a small seeded synthetic
tick series with no external data — a one-command, reproducible smoke run.

The heavy imports (pandas IO, the engine) happen inside the command functions
so that importing this module — for ``--help`` or testing — never pulls in the
full stack. ``typer`` is used when available and falls back to ``argparse`` so
the CLI works with only the core dependencies installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Commands (heavy imports are local)
# --------------------------------------------------------------------------- #
def cmd_demo(seed: int, out: str) -> dict[str, Any]:
    """Run the full pipeline on a seeded synthetic tick series (no external data)."""
    from backtest.demo import run_demo

    result = run_demo(seed=seed, out_dir=out)
    print(json.dumps(result, indent=2, default=str))
    return result


def cmd_bars(config: str, ticks: str, out: str) -> None:
    """Resample a raw tick CSV to OHLCV bars and write Parquet."""
    import pandas as pd

    from backtest.bars import resample_ohlcv

    cfg = _load_config(config)
    dcfg = cfg["data"]
    bcfg = cfg["bars"]

    raw = pd.read_csv(ticks)
    raw[dcfg["time_column"]] = pd.to_datetime(
        raw[dcfg["time_column"]], unit=dcfg.get("time_unit", "ms")
    )
    bars = resample_ohlcv(
        raw,
        bcfg["rule"],
        ts=dcfg["time_column"],
        price=dcfg["price_column"],
        size=dcfg["size_column"],
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out)
    print(f"wrote {len(bars)} bars -> {out}")


def cmd_run(config: str, bars: str, out: str) -> dict[str, Any]:
    """Run the no-look-ahead backtest from a bar Parquet; write the equity curve."""
    import pandas as pd

    from backtest.engine import backtest
    from backtest.indicators import sma
    from backtest.performance import max_drawdown, sharpe, total_return

    cfg = _load_config(config)
    icfg = cfg["indicators"]
    ccfg = cfg["costs"]
    pcfg = cfg["performance"]

    df = pd.read_parquet(bars)
    close = df["close"].astype(float)

    fast = sma(close, icfg["fast"])
    slow = sma(close, icfg["slow"])
    signal = (fast > slow).astype(float)  # 1 long when fast > slow, else flat

    equity = backtest(
        close,
        signal,
        fee_bps=ccfg["fee_bps"],
        slippage_bps=ccfg["slippage_bps"],
    )
    rets = equity.pct_change().fillna(0.0)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    equity.to_frame("equity").to_parquet(out_dir / "equity.parquet")

    summary = {
        "n_bars": int(len(close)),
        "total_return": total_return(equity.to_numpy()),
        "sharpe": sharpe(rets.to_numpy(), pcfg["periods_per_year"]),
        "max_drawdown": max_drawdown(equity.to_numpy()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def cmd_report(config: str, equity: str, out: str) -> dict[str, Any]:
    """Produce the performance report from a saved equity curve."""
    import pandas as pd

    from backtest.performance import max_drawdown, sharpe, total_return

    cfg = _load_config(config)
    pcfg = cfg["performance"]

    eq = pd.read_parquet(equity)["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)
    summary = {
        "total_return": total_return(eq.to_numpy()),
        "sharpe": sharpe(rets.to_numpy(), pcfg["periods_per_year"]),
        "max_drawdown": max_drawdown(eq.to_numpy()),
    }
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# Entry point: prefer typer, fall back to argparse
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    try:
        import typer
    except ImportError:
        return _argparse_main(argv)

    app = typer.Typer(add_completion=False, help=__doc__)

    @app.command()
    def demo(seed: int = 0, out: str = "outputs"):
        """Run the full pipeline on a seeded synthetic tick series."""
        cmd_demo(seed, out)

    @app.command()
    def bars(config: str = "config/strategy.yaml", ticks: str = ..., out: str = ...):
        """Resample raw ticks to OHLCV bars."""
        cmd_bars(config, ticks, out)

    @app.command()
    def run(
        config: str = "config/strategy.yaml",
        bars: str = ...,
        out: str = "outputs",
    ):
        """Run the no-look-ahead backtest."""
        cmd_run(config, bars, out)

    @app.command()
    def report(
        config: str = "config/strategy.yaml", equity: str = ..., out: str = "outputs"
    ):
        """Write the performance report from an equity curve."""
        cmd_report(config, equity, out)

    app(argv)
    return 0


def _argparse_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="backtest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo", help="Run the full pipeline on a seeded synthetic tick series."
    )
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out", default="outputs")

    p_bars = sub.add_parser("bars", help="Resample raw ticks to OHLCV bars.")
    p_bars.add_argument("--config", default="config/strategy.yaml")
    p_bars.add_argument("--ticks", required=True)
    p_bars.add_argument("--out", required=True)

    p_run = sub.add_parser("run", help="Run the no-look-ahead backtest.")
    p_run.add_argument("--config", default="config/strategy.yaml")
    p_run.add_argument("--bars", required=True)
    p_run.add_argument("--out", default="outputs")

    p_rep = sub.add_parser("report", help="Performance report from an equity curve.")
    p_rep.add_argument("--config", default="config/strategy.yaml")
    p_rep.add_argument("--equity", required=True)
    p_rep.add_argument("--out", default="outputs")

    args = parser.parse_args(argv)
    if args.command == "demo":
        cmd_demo(args.seed, args.out)
    elif args.command == "bars":
        cmd_bars(args.config, args.ticks, args.out)
    elif args.command == "run":
        cmd_run(args.config, args.bars, args.out)
    elif args.command == "report":
        cmd_report(args.config, args.equity, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
