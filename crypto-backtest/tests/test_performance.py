"""Known-answer tests for the performance analytics.

* max_drawdown([100, 120, 90, 150]) = 0.25: peak 120, trough 90,
  (120 - 90) / 120 = 0.25.
* sharpe of a zero-variance return series = 0.0 (defined, not a divide error).
* total_return([100, 150]) = 0.5.
"""

from __future__ import annotations

import pytest

from backtest.performance import max_drawdown, sharpe, total_return


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
