"""Known-answer tests for the performance analytics.

* max_drawdown([100, 120, 90, 150]) = 0.25: peak 120, trough 90,
  (120 - 90) / 120 = 0.25.
* sharpe of a zero-variance return series = 0.0 (defined, not a divide error).
* total_return([100, 150]) = 0.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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


def test_max_drawdown_quarter() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 150.0]) == 0.25


def test_max_drawdown_monotone_is_zero() -> None:
    assert max_drawdown([1.0, 2.0, 3.0, 4.0]) == 0.0


def test_sharpe_zero_variance_is_zero() -> None:
    assert sharpe([0.0, 0.0, 0.0, 0.0]) == 0.0
    assert sharpe([0.01, 0.01, 0.01]) == 0.0  # constant nonzero returns


def test_sharpe_known_value() -> None:
    # Returns alternating +/- a with mean 0 -> Sharpe 0 regardless of scaling.
    assert sharpe([0.02, -0.02, 0.02, -0.02], periods_per_year=252.0) == 0.0
    # Positive constant-mean series: mean=0.01, std of [0.0,0.02]=0.01,
    # annualised by sqrt(4): 2 * 0.01 / 0.01 = 2.0.
    assert sharpe([0.0, 0.02], periods_per_year=4.0) == pytest.approx(2.0, abs=1e-12)


def test_total_return_known() -> None:
    assert total_return([100.0, 150.0]) == 0.5
    assert total_return([100.0, 110.0, 121.0]) == pytest.approx(0.21, abs=1e-12)


def test_total_return_degenerate() -> None:
    assert total_return([100.0]) == 0.0
    assert total_return([]) == 0.0


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------


def test_sortino_no_downside_is_inf() -> None:
    # All non-negative returns with a positive mean -> downside std 0,
    # mean > 0 -> ratio is +inf (documented sentinel).
    assert sortino([0.0, 0.01, 0.02, 0.0]) == math.inf


def test_sortino_flat_series_is_zero() -> None:
    # Zero mean and zero downside -> 0.0, not nan.
    assert sortino([0.0, 0.0, 0.0]) == 0.0


def test_sortino_hand_value() -> None:
    # returns [0.02, -0.01]: mean = 0.005.
    # downside = [0, -0.01]; downside_var = (0 + 0.0001)/2 = 0.00005;
    # downside_std = sqrt(0.00005). annualise sqrt(4)=2.
    # sortino = 2 * 0.005 / sqrt(0.00005).
    expected = 2.0 * 0.005 / math.sqrt(0.00005)
    assert sortino([0.02, -0.01], periods_per_year=4.0) == pytest.approx(
        expected, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Calmar
# ---------------------------------------------------------------------------


def test_calmar_hand_value() -> None:
    # equity [100, 120, 90, 150] over 3 periods, periods_per_year=3 -> CAGR =
    # (150/100)^(3/3) - 1 = 0.5. max_drawdown = 0.25. calmar = 0.5/0.25 = 2.0.
    assert calmar([100.0, 120.0, 90.0, 150.0], periods_per_year=3.0) == pytest.approx(
        2.0, abs=1e-12
    )


def test_calmar_no_drawdown_is_inf() -> None:
    # Monotone-up curve has zero drawdown and positive growth -> +inf.
    assert calmar([1.0, 2.0, 3.0, 4.0]) == math.inf


def test_calmar_degenerate() -> None:
    assert calmar([100.0]) == 0.0
    assert calmar([]) == 0.0


# ---------------------------------------------------------------------------
# win_rate / turnover / exposure
# ---------------------------------------------------------------------------


def test_win_rate_known() -> None:
    # 2 of 4 strictly positive (zero is not a win).
    assert win_rate([0.1, -0.2, 0.0, 0.3]) == 0.5


def test_win_rate_empty_is_zero() -> None:
    assert win_rate([]) == 0.0


def test_turnover_known() -> None:
    # positions [0, 1, 1, 0]; changes vs flat start [0, 1, 0, 1]; mean 0.5.
    assert turnover([0.0, 1.0, 1.0, 0.0]) == 0.5


def test_turnover_empty_is_zero() -> None:
    assert turnover([]) == 0.0
    assert turnover(np.array([])) == 0.0


def test_exposure_known() -> None:
    # 2 of 4 bars hold a non-zero position.
    assert exposure([0.0, 1.0, 1.0, 0.0]) == 0.5
    # short counts as exposed too.
    assert exposure([-1.0, 0.0]) == 0.5


def test_exposure_empty_is_zero() -> None:
    assert exposure([]) == 0.0
