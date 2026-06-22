"""Known-answer tests for the backtest engine, focused on no-look-ahead.

The engine executes a signal from bar ``t`` on bar ``t + 1`` (positions are
``signals.shift(1)``). The tests below prove (a) a signal cannot affect the bar
on which it is generated, and (b) a full equity path with one fee charge
matches a hand calculation.

Hand calc for the fee case:

  prices  = [100, 110, 121, 121]   -> bar returns [_, 0.10, 0.10, 0.00]
  signals = [1, 1, 1, 1]  (always long)
  positions = shift(1)    = [0, 1, 1, 1]   (flat over the first bar)
  fee_bps + slippage_bps  = 10 + 5 = 15 bps = 0.0015
  turnover (|dposition|)  = [0, 1, 0, 0]   -> one entry, charged once
  net returns = [0, 0.10 - 0.0015, 0.10, 0.00] = [0, 0.0985, 0.10, 0.00]
  equity (cumprod, start 1.0):
      1.0
      1.0 * 1.0985            = 1.0985
      1.0985 * 1.10           = 1.20835
      1.20835 * 1.00          = 1.20835
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import backtest, positions_from_signals


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype=float)


def test_no_look_ahead_signal_acts_on_next_bar() -> None:
    """A signal on bar t must not change equity until bar t+1 (no fees)."""
    prices = _series([100.0, 110.0, 121.0, 90.0])
    signals = _series([0.0, 1.0, 0.0, 0.0])  # go long on bar 1

    equity = backtest(prices, signals, fee_bps=0.0, slippage_bps=0.0)

    # Position over each bar is the previous bar's signal.
    pos = positions_from_signals(signals)
    assert pos.tolist() == [0.0, 0.0, 1.0, 0.0]

    # Bar 1 carries the signal but flat position -> equity unchanged at bar 1.
    assert equity.iloc[0] == 1.0
    assert equity.iloc[1] == 1.0
    # The long position is held over bar 2's +10% return: equity becomes 1.1.
    assert equity.iloc[2] == pytest.approx(1.1, abs=1e-12)
    # Back to flat for bar 3 -> the -25.6% move does not touch equity.
    assert equity.iloc[3] == pytest.approx(1.1, abs=1e-12)


def test_constant_long_equity_with_one_fee() -> None:
    prices = _series([100.0, 110.0, 121.0, 121.0])
    signals = _series([1.0, 1.0, 1.0, 1.0])

    equity = backtest(prices, signals, fee_bps=10.0, slippage_bps=5.0)

    expected = [1.0, 1.0985, 1.20835, 1.20835]
    assert np.allclose(equity.to_numpy(), expected, atol=1e-12)


def test_zero_signal_is_flat_equity() -> None:
    prices = _series([100.0, 110.0, 90.0, 130.0])
    signals = _series([0.0, 0.0, 0.0, 0.0])
    equity = backtest(prices, signals, fee_bps=10.0, slippage_bps=5.0)
    assert np.allclose(equity.to_numpy(), [1.0, 1.0, 1.0, 1.0])
