"""crypto-backtest: rigorous, no-look-ahead event-driven backtesting.

The differentiator in this project is rigour over returns. The core modules
are deliberately small and pure (numpy/pandas/stdlib only), so they are always
importable and are pinned down by hand-derived known-answer tests:

* :mod:`backtest.bars` resamples a raw tick stream to OHLCV bars.
* :mod:`backtest.indicators` computes SMA / EMA / Wilder RSI / rolling vol plus
  MACD, Bollinger bands, and Wilder ATR.
* :mod:`backtest.integrity` finds gaps, duplicates, and summarises data health.
* :mod:`backtest.engine` runs the backtest. A signal generated on bar ``t`` is
  executed at bar ``t + 1`` (signals are shifted by one), so the engine cannot
  peek at information it would not have had in real time. Fees and slippage are
  charged on every position change.
* :mod:`backtest.performance` reports total return, Sharpe, max drawdown,
  Sortino, Calmar, win rate, turnover, and exposure.
* :mod:`backtest.validation` provides walk-forward splits and a fee/slippage
  sensitivity sweep over the real engine.

Large multi-symbol resampling (Polars / Spark) lives in
:mod:`backtest.scale_pipeline` behind lazy imports and is intentionally *not*
imported here, so this package and its tests run on numpy/pandas alone.
"""

from __future__ import annotations

from backtest.bars import resample_ohlcv
from backtest.engine import backtest, positions_from_signals
from backtest.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rolling_vol,
    rsi,
    sma,
)
from backtest.integrity import find_duplicates, find_gaps, summarize_integrity
from backtest.performance import (
    calmar,
    exposure,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    turnover,
    win_rate,
)
from backtest.validation import sensitivity_sweep, walk_forward_splits

__all__ = [
    "resample_ohlcv",
    "sma",
    "ema",
    "rsi",
    "rolling_vol",
    "macd",
    "bollinger_bands",
    "atr",
    "find_gaps",
    "find_duplicates",
    "summarize_integrity",
    "backtest",
    "positions_from_signals",
    "total_return",
    "sharpe",
    "max_drawdown",
    "sortino",
    "calmar",
    "win_rate",
    "turnover",
    "exposure",
    "walk_forward_splits",
    "sensitivity_sweep",
    "__version__",
]

__version__ = "0.1.0"
