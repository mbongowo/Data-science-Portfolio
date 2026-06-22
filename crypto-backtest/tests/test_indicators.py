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

from backtest.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rolling_vol,
    rsi,
    sma,
)


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


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def test_macd_constant_series_is_all_zero() -> None:
    # A constant series has identical fast/slow EMAs, so macd == signal ==
    # histogram == 0 everywhere (hand-derived: ema(c) == c for any span).
    out = macd([5.0] * 30, fast=12, slow=26, signal=9)
    assert list(out.columns) == ["macd", "signal", "histogram"]
    assert np.allclose(out["macd"].to_numpy(), 0.0)
    assert np.allclose(out["signal"].to_numpy(), 0.0)
    assert np.allclose(out["histogram"].to_numpy(), 0.0)


def test_macd_hand_value_small_spans() -> None:
    # x = [1, 2, 3], fast=1 (alpha=1 -> ema == x), slow=3 (alpha=0.5),
    # signal=1 (alpha=1 -> signal == macd line, histogram == 0).
    #   ema_slow: e0=1; e1=.5*2+.5*1=1.5; e2=.5*3+.5*1.5=2.25
    #   macd line = x - ema_slow = [0, 0.5, 0.75]
    out = macd([1.0, 2.0, 3.0], fast=1, slow=3, signal=1)
    assert np.allclose(out["macd"].to_numpy(), [0.0, 0.5, 0.75])
    # signal span 1 -> follows macd exactly -> histogram all zero.
    assert np.allclose(out["signal"].to_numpy(), [0.0, 0.5, 0.75])
    assert np.allclose(out["histogram"].to_numpy(), 0.0)


def test_macd_rejects_fast_ge_slow() -> None:
    with pytest.raises(ValueError):
        macd([1.0, 2.0, 3.0], fast=26, slow=12)


# ---------------------------------------------------------------------------
# Bollinger bands
# ---------------------------------------------------------------------------


def test_bollinger_hand_value() -> None:
    # x = [1,2,3,4,5], n=3, k=2. Last window [3,4,5]: mean 4, population
    # variance ((3-4)^2+(4-4)^2+(5-4)^2)/3 = 2/3, sigma = sqrt(2/3).
    out = bollinger_bands([1.0, 2.0, 3.0, 4.0, 5.0], n=3, k=2.0)
    assert list(out.columns) == ["middle", "upper", "lower"]
    assert math.isnan(out["middle"].iloc[0])
    assert math.isnan(out["middle"].iloc[1])
    sigma = math.sqrt(2.0 / 3.0)
    assert out["middle"].iloc[4] == pytest.approx(4.0, abs=1e-12)
    assert out["upper"].iloc[4] == pytest.approx(4.0 + 2.0 * sigma, abs=1e-12)
    assert out["lower"].iloc[4] == pytest.approx(4.0 - 2.0 * sigma, abs=1e-12)


def test_bollinger_constant_series_collapses() -> None:
    # Constant series -> sigma 0 -> upper == middle == lower == price.
    out = bollinger_bands([7.0, 7.0, 7.0, 7.0], n=2, k=2.0)
    defined = out.iloc[1:]
    assert np.allclose(defined["middle"].to_numpy(), 7.0)
    assert np.allclose(defined["upper"].to_numpy(), 7.0)
    assert np.allclose(defined["lower"].to_numpy(), 7.0)


# ---------------------------------------------------------------------------
# ATR (Wilder)
# ---------------------------------------------------------------------------


def test_atr_hand_value_n1() -> None:
    # n=1 -> alpha=1 -> ATR equals the true range each bar.
    #   bar0: no prior close -> TR = high-low = 11 - 9 = 2
    #   bar1: prev_close=10; TR = max(12-10, |12-10|, |10-10|) = 2
    #   bar2: prev_close=11; TR = max(13-11, |13-11|, |11-11|) = 2
    high = [11.0, 12.0, 13.0]
    low = [9.0, 10.0, 11.0]
    close = [10.0, 11.0, 12.0]
    out = atr(high, low, close, n=1)
    assert np.allclose(out.to_numpy(), [2.0, 2.0, 2.0])


def test_atr_constant_prices_is_zero() -> None:
    # Flat high==low==close -> every true range is 0 -> ATR is 0 throughout.
    out = atr([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], n=2)
    assert np.allclose(out.to_numpy(), 0.0)


def test_atr_gap_uses_prev_close() -> None:
    # A gap-up bar: prev close 10, next bar high 20, low 18.
    #   TR = max(20-18, |20-10|, |18-10|) = max(2, 10, 8) = 10.
    out = atr([10.0, 20.0], [10.0, 18.0], [10.0, 19.0], n=1)
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-12)  # bar0 high==low
    assert out.iloc[1] == pytest.approx(10.0, abs=1e-12)
