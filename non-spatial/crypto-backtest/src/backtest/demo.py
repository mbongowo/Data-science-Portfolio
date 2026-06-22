"""A one-command, end-to-end demo on a small seeded synthetic tick series.

This module exists so the README can quote **real, reproducible** numbers
instead of placeholders. It synthesises a tiny intraday trade (tick) stream
from a seeded geometric-Brownian-motion price path and then drives the *actual*
core — integrity checks, resampling, indicators, the no-look-ahead engine, and
the performance analytics — exactly as a real run would. It depends on nothing
beyond numpy / pandas / pyyaml + stdlib, so it runs anywhere, including CI, in
well under a second.

Honest framing: the price path is synthetic and seeded. The point is to
exercise the real pipeline deterministically and report defensible metrics
(execution delayed to ``t + 1``, fees and slippage on every position change),
**not** to claim a real-world edge. The same ``resample -> indicators ->
backtest -> performance`` path runs on real Binance tick dumps via the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.bars import resample_ohlcv
from backtest.engine import backtest
from backtest.indicators import rsi, sma
from backtest.integrity import summarize_integrity
from backtest.performance import max_drawdown, sharpe, total_return

# Demo parameters. Fixed here (not read from YAML) so the committed metrics in
# the tests are pinned to exactly this configuration.
_N_TICKS = 60_000
_START_PRICE = 30_000.0
_MU = 0.0  # per-tick drift of the GBM (zero: no built-in edge)
_SIGMA = 0.0008  # per-tick volatility of the GBM log-returns
_BAR_RULE = "1min"
_TICK_FREQ = "1s"  # one synthetic trade per second
_FAST = 10  # fast SMA window (bars)
_SLOW = 30  # slow SMA window (bars)
_RSI_N = 14
_FEE_BPS = 10.0
_SLIPPAGE_BPS = 5.0
_PERIODS_PER_YEAR = 525_600  # 1-minute bars, 24/7 crypto market


def _synthesize_ticks(seed: int) -> pd.DataFrame:
    """Build a deterministic synthetic tick frame ``(ts, price, size)``.

    The price follows a seeded geometric Brownian motion (multiplicative
    log-returns), so prices stay positive and the path is realistic enough to
    drive resampling and a crossover strategy. One trade is emitted per second.
    """
    rng = np.random.default_rng(seed)
    log_rets = _MU + _SIGMA * rng.standard_normal(_N_TICKS)
    price = _START_PRICE * np.exp(np.cumsum(log_rets))
    size = rng.uniform(0.001, 0.5, size=_N_TICKS)
    ts = pd.date_range("2024-01-01", periods=_N_TICKS, freq=_TICK_FREQ)
    return pd.DataFrame({"ts": ts, "price": price, "size": size})


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the full pipeline on a seeded synthetic tick series.

    Steps, all on the real core:

    1. Synthesize ticks from a seeded GBM price path.
    2. Run integrity checks on the resulting bar index (gaps / duplicates).
    3. Resample ticks to 1-minute OHLCV bars.
    4. Compute fast/slow SMA and Wilder RSI (strictly trailing).
    5. Generate a long/flat SMA-crossover signal: long (1) when fast SMA >
       slow SMA, else flat (0). Both windows are trailing, so the signal on a
       bar uses only that bar and earlier.
    6. Run the no-look-ahead engine (signal on bar ``t`` executes on ``t + 1``)
       with realistic fees and slippage.
    7. Compute performance (total return, Sharpe, max drawdown) and a
       buy-and-hold baseline for honest comparison.

    Artifacts ``bars.csv``, ``equity_curve.csv`` and ``summary.json`` are
    written to ``out_dir``.

    Returns
    -------
    dict
        ``num_ticks``, ``num_bars``, ``num_trades``, ``total_return``,
        ``sharpe``, ``max_drawdown`` and ``buy_and_hold_return``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic ticks.
    ticks = _synthesize_ticks(seed)

    # 2. Resample ticks -> OHLCV bars (the real resampler).
    bars = resample_ohlcv(ticks, _BAR_RULE, ts="ts", price="price", size="size")
    close = bars["close"].astype(float)

    # 3. Integrity of the bar series (gaps / duplicates / coverage).
    integrity = summarize_integrity(bars, _BAR_RULE)

    # 4. Indicators (strictly trailing).
    fast = sma(close, _FAST)
    slow = sma(close, _SLOW)
    bars["sma_fast"] = fast
    bars["sma_slow"] = slow
    bars["rsi"] = rsi(close, _RSI_N)

    # 5. SMA-crossover signal: long when fast > slow, else flat. NaN windows
    #    (before a full slow window exists) are treated as flat.
    signal = (fast > slow).astype(float)

    # 6. No-look-ahead backtest with realistic costs.
    equity = backtest(
        close,
        signal,
        fee_bps=_FEE_BPS,
        slippage_bps=_SLIPPAGE_BPS,
    )
    rets = equity.pct_change().fillna(0.0)

    # Number of executed trades = position changes (signal acts on t+1).
    position = signal.shift(1).fillna(0.0)
    num_trades = int((position.diff().fillna(position.iloc[0]).abs() > 0).sum())

    # 7. Buy-and-hold baseline over the same bars (no costs, always long).
    bh_equity = close / close.iloc[0]

    result = {
        "num_ticks": int(len(ticks)),
        "num_bars": int(len(bars)),
        "num_trades": num_trades,
        "total_return": total_return(equity.to_numpy()),
        "sharpe": sharpe(rets.to_numpy(), _PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown(equity.to_numpy()),
        "buy_and_hold_return": total_return(bh_equity.to_numpy()),
    }

    # Artifacts.
    bars.to_csv(out_path / "bars.csv", index_label="ts")
    equity.to_frame("equity").to_csv(out_path / "equity_curve.csv", index_label="ts")
    summary = {
        "seed": int(seed),
        "assumptions": {
            "bar_rule": _BAR_RULE,
            "fast_sma": _FAST,
            "slow_sma": _SLOW,
            "rsi_window": _RSI_N,
            "fee_bps": _FEE_BPS,
            "slippage_bps": _SLIPPAGE_BPS,
            "periods_per_year": _PERIODS_PER_YEAR,
            "no_look_ahead": "signal on bar t executes on bar t+1",
            "price_path": "seeded synthetic geometric Brownian motion",
        },
        "integrity": integrity,
        "metrics": result,
    }
    with open(out_path / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return result


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), default=str, indent=2))
