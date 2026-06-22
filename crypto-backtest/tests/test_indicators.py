"""Known-answer tests for the indicators (pure numpy/pandas).

Each expected value is computed by hand:

* RSI of a strictly increasing series has no down moves, so the average loss is
  0, RS is infinite, and RSI = 100 exactly (Wilder).
* SMA([1,2,3,4,5], 3): the last window is (3+4+5)/3 = 4.0; the window ending at
  index 2 is (1+2+3)/3 = 2.0; the first two positions are NaN.
* rolling_vol of returns [0.1, 0.2, 0.3] with window 2 (population std):
  at index 2 the window is [0.2, 0.3], mean 0.25, variance
  ((0.2-0.25)^2 + (0.3-0.25)^2)/2 = 0.0025, std = 0.05.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.indicators import ema, rolling_vol, rsi, sma


def test_rsi_strictly_increasing_is_100() -> None:
    prices = pd.Series([float(i) for i in range(1, 21)])
    out = rsi(prices, 14)
    # Every defined RSI value on a monotone-up series is 100.
    assert out.iloc[1:].tolist() == [100.0] * (len(prices) - 1)


def test_rsi_strictly_decreasing_is_0() -> None:
    prices = pd.Series([float(i) for i in range(20, 0, -1)])
    out = rsi(prices, 14)
    assert out.iloc[1:].tolist() == [0.0] * (len(prices) - 1)


def test_sma_hand_value() -> None:
    out = sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_rolling_vol_hand_value() -> None:
    out = rolling_vol([0.1, 0.2, 0.3], 2)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.05, abs=1e-12)
    assert out.iloc[2] == pytest.approx(0.05, abs=1e-12)


def test_ema_is_recursive_seeded_from_first() -> None:
    # adjust=False: e0 = x0; e1 = alpha*x1 + (1-alpha)*e0, alpha = 2/(n+1).
    x = [1.0, 2.0, 3.0]
    out = ema(x, 1)  # alpha = 2/2 = 1 -> EMA follows the series exactly
    assert np.allclose(out.to_numpy(), x)
