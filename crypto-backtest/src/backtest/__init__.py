"""crypto-backtest: rigorous, no-look-ahead event-driven backtesting.

The differentiator in this project is rigour over returns. The core modules
are deliberately small and pure (numpy/pandas/stdlib only), so they are always
importable and are pinned down by hand-derived known-answer tests:

* :mod:`backtest.bars` resamples a raw tick stream to OHLCV bars.
* :mod:`backtest.indicators` computes SMA / EMA / Wilder RSI / rolling vol.
* :mod:`backtest.integrity` finds gaps, duplicates, and summarises data health.
* :mod:`backtest.engine` runs the backtest. A signal generated on bar ``t`` is
  executed at bar ``t + 1`` (signals are shifted by one), so the engine cannot
  peek at information it would not have had in real time. Fees and slippage are
  charged on every position change.
* :mod:`backtest.performance` reports total return, Sharpe, and max drawdown.

Large multi-symbol resampling (Polars / Spark) lives in
:mod:`backtest.scale_pipeline` behind lazy imports and is intentionally *not*
imported here, so this package and its tests run on numpy/pandas alone.
"""

from __future__ import annotations

from backtest.bars import resample_ohlcv
from backtest.engine import backtest
from backtest.indicators import ema, rolling_vol, rsi, sma
from backtest.integrity import find_duplicates, find_gaps, summarize_integrity
from backtest.performance import max_drawdown, sharpe, total_return

__all__ = [
    "resample_ohlcv",
    "sma",
    "ema",
    "rsi",
    "rolling_vol",
    "find_gaps",
    "find_duplicates",
    "summarize_integrity",
    "backtest",
    "total_return",
    "sharpe",
    "max_drawdown",
    "__version__",
]

__version__ = "0.1.0"
