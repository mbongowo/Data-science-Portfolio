"""Hand-derived known-answer tests for the robust trend statistics."""

from __future__ import annotations

import numpy as np
import pytest

from disturb.trend import mann_kendall, theil_sen_slope


def test_theil_sen_perfect_line_returns_exact_slope():
    # y = 3 + 2.5 t: every pairwise slope is exactly 2.5.
    t = np.arange(20, dtype=float)
    y = 3.0 + 2.5 * t
    assert theil_sen_slope(y, t) == pytest.approx(2.5, abs=1e-12)


def test_theil_sen_default_time_axis():
    # With no t, the axis is 0..n-1; a slope-of-2 ramp recovers 2.
    y = np.arange(0.0, 20.0, 2.0)  # step of 2 per index
    assert theil_sen_slope(y) == pytest.approx(2.0, abs=1e-12)


def test_theil_sen_robust_to_outlier():
    # One wild point must not move the median slope off the true line.
    t = np.arange(21, dtype=float)
    y = 1.0 + 0.5 * t
    y[10] += 100.0  # gross outlier
    assert theil_sen_slope(y, t) == pytest.approx(0.5, abs=1e-9)


def test_theil_sen_too_short_is_nan():
    assert np.isnan(theil_sen_slope(np.array([1.0])))


def test_theil_sen_constant_series_zero_slope():
    assert theil_sen_slope(np.full(10, 0.7)) == pytest.approx(0.0, abs=1e-12)


def test_mann_kendall_strictly_increasing_is_positive():
    res = mann_kendall(np.arange(30, dtype=float))
    assert res.trend == "increasing"
    # All n(n-1)/2 pairs are positive: S = 30*29/2 = 435.
    assert res.s == 435
    assert res.z > 0
    assert res.p_value < 0.05


def test_mann_kendall_strictly_decreasing_is_negative():
    res = mann_kendall(np.arange(30, 0, -1, dtype=float))
    assert res.trend == "decreasing"
    assert res.s == -435
    assert res.p_value < 0.05


def test_mann_kendall_flat_series_no_trend():
    res = mann_kendall(np.full(30, 0.5))
    assert res.trend == "no trend"
    assert res.s == 0
    assert res.z == 0.0
    assert res.p_value == pytest.approx(1.0)


def test_mann_kendall_too_short_is_no_trend():
    res = mann_kendall(np.array([1.0, 2.0]))
    assert res.trend == "no trend"


def test_mann_kendall_noisy_uptrend_detected():
    rng = np.random.default_rng(0)
    n = 60
    y = 0.01 * np.arange(n) + rng.normal(0, 0.02, n)
    res = mann_kendall(y)
    assert res.trend == "increasing"
    assert res.p_value < 0.05
