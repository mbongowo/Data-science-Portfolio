"""A synthetic step drop must be flagged at the correct index/date."""

from __future__ import annotations

import numpy as np
import pytest

from disturb.decompose import harmonic_decompose
from disturb.detect import detect_breakpoint


def test_step_drop_detected_at_known_index():
    rng = np.random.default_rng(0)
    n = 100
    break_at = 60  # disturbance occurs after sample 60
    y = rng.normal(0.0, 0.02, size=n)
    y[break_at + 1 :] -= 0.4  # sharp NDVI drop (fire/clearing)

    bp = detect_breakpoint(y, min_segment=5, threshold=1.0)

    assert bp.detected
    assert abs(bp.index - break_at) <= 2  # within tolerance
    assert bp.magnitude < -0.3  # a real drop, sign negative


def test_break_date_is_reported():
    rng = np.random.default_rng(1)
    n = 80
    break_at = 50
    times = np.arange("2019-01-01", n * 16, 16, dtype="datetime64[D]")
    assert times.size == n
    y = rng.normal(0.5, 0.02, size=n)
    y[break_at + 1 :] -= 0.5

    bp = detect_breakpoint(y, times=times, min_segment=5)

    assert bp.detected
    expected_date = times[break_at + 1]
    # The reported date should be within ~2 samples (32 days) of truth.
    assert abs((bp.date - expected_date) / np.timedelta64(1, "D")) <= 40


def test_no_break_in_flat_series_low_score():
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 0.02, size=100)
    bp = detect_breakpoint(y, min_segment=5, threshold=5.0)
    # A flat noisy series should not clear a high threshold.
    assert not bp.detected


def test_detect_on_residual_after_decompose():
    """End-to-end: deseasonalise, then detect the injected disturbance."""
    rng = np.random.default_rng(3)
    period = 365.25
    t = np.arange(0, 5 * 365, 16, dtype=float)
    n = t.size
    break_at = n * 3 // 5
    seasonal = 0.25 * np.sin(2 * np.pi * t / period)
    y = 0.6 + seasonal + rng.normal(0.0, 0.01, size=n)
    y[break_at + 1 :] -= 0.35

    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    bp = detect_breakpoint(fit.residual, times=None, min_segment=5)

    assert bp.detected
    # The break index is recovered even though the residual magnitude is
    # diluted: the linear trend term partly absorbs a sustained step, so we
    # check the location and the (negative) sign rather than the full depth.
    assert abs(bp.index - break_at) <= 3
    assert bp.magnitude < 0.0


def test_rise_has_positive_magnitude():
    """An upward step (e.g. regrowth) yields a positive magnitude."""
    rng = np.random.default_rng(4)
    n = 100
    y = rng.normal(0.0, 0.02, size=n)
    y[51:] += 0.4
    bp = detect_breakpoint(y, min_segment=5, threshold=1.0)
    assert bp.detected
    assert bp.magnitude > 0.3


def test_drop_has_negative_magnitude():
    """A downward step yields a negative magnitude of the expected size."""
    rng = np.random.default_rng(5)
    n = 100
    y = rng.normal(0.0, 0.02, size=n)
    y[51:] -= 0.4
    bp = detect_breakpoint(y, min_segment=5, threshold=1.0)
    assert bp.detected
    assert bp.magnitude == pytest.approx(-0.4, abs=0.05)


def test_single_outlier_does_not_clear_threshold():
    """One spike must score below a sustained step at the same threshold."""
    rng = np.random.default_rng(6)
    n = 100
    spike = rng.normal(0.0, 0.02, size=n)
    spike[50] += 1.0  # one bad sample (uncaught cloud)
    bp_spike = detect_breakpoint(spike, min_segment=5, threshold=1.0)

    step = rng.normal(0.0, 0.02, size=n)
    step[51:] -= 0.4  # genuine sustained disturbance
    bp_step = detect_breakpoint(step, min_segment=5, threshold=1.0)

    assert not bp_spike.detected
    assert bp_step.detected
    assert bp_step.score > bp_spike.score


def test_detection_within_tolerance_window():
    """The reported index sits within a few samples of the true break."""
    rng = np.random.default_rng(8)
    n = 120
    break_at = 73
    y = rng.normal(0.0, 0.015, size=n)
    y[break_at + 1 :] -= 0.5
    bp = detect_breakpoint(y, min_segment=6, threshold=1.0)
    assert bp.detected
    assert abs(bp.index - break_at) <= 2


def test_too_short_series_raises():
    y = np.zeros(6)
    with pytest.raises(ValueError):
        detect_breakpoint(y, min_segment=5)


def test_nan_tolerant_detection():
    """NaN gaps do not crash the scan and the break is still found."""
    rng = np.random.default_rng(9)
    n = 100
    break_at = 60
    y = rng.normal(0.0, 0.02, size=n)
    y[break_at + 1 :] -= 0.4
    y[::9] = np.nan  # cloud gaps
    bp = detect_breakpoint(y, min_segment=5, threshold=1.0)
    assert bp.detected
    assert abs(bp.index - break_at) <= 4
    assert bp.magnitude < 0.0
