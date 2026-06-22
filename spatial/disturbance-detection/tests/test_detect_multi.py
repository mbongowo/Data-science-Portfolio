"""Known-answer tests for binary segmentation and recovery-time."""

from __future__ import annotations

import numpy as np

from disturb.detect import detect_breakpoints_binseg, recovery_time


def test_binseg_finds_two_planted_steps():
    rng = np.random.default_rng(0)
    n = 120
    y = rng.normal(0.0, 0.02, size=n)
    y[40:] -= 0.4  # first step down
    y[80:] += 0.6  # second step up
    bps = detect_breakpoints_binseg(
        y, max_breaks=3, threshold=1.0, min_segment=5
    )
    assert len(bps) == 2
    idx = sorted(b.index for b in bps)
    assert abs(idx[0] - 39) <= 2
    assert abs(idx[1] - 79) <= 2
    # Recomputed magnitudes carry the right signs.
    by_index = {b.index: b for b in bps}
    assert by_index[idx[0]].magnitude < 0
    assert by_index[idx[1]].magnitude > 0


def test_binseg_respects_max_breaks():
    rng = np.random.default_rng(1)
    n = 160
    y = rng.normal(0.0, 0.02, size=n)
    y[40:] -= 0.4
    y[80:] += 0.6
    y[120:] -= 0.5
    bps = detect_breakpoints_binseg(
        y, max_breaks=2, threshold=1.0, min_segment=5
    )
    assert len(bps) == 2


def test_binseg_no_break_in_flat_series():
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 0.02, size=100)
    bps = detect_breakpoints_binseg(
        y, max_breaks=3, threshold=5.0, min_segment=5
    )
    assert bps == []


def test_binseg_attaches_dates():
    rng = np.random.default_rng(3)
    n = 120
    times = np.arange("2019-01-01", n, dtype="datetime64[D]")
    y = rng.normal(0.0, 0.02, size=n)
    y[60:] -= 0.5
    bps = detect_breakpoints_binseg(
        y, times=times, max_breaks=3, threshold=1.0, min_segment=5
    )
    assert len(bps) >= 1
    assert all(b.date is not None for b in bps)


def test_recovery_time_recovers():
    # Pre-break level 1.0, drops to 0.3 for 5 samples, returns to 1.0.
    y = np.concatenate([np.full(20, 1.0), np.full(5, 0.3), np.full(20, 1.0)])
    # Break sits after index 19 (the last pre-break sample).
    assert recovery_time(y, 19, tolerance=0.1) == 6


def test_recovery_time_never_recovers():
    y = np.concatenate([np.full(20, 1.0), np.full(20, 0.3)])
    assert recovery_time(y, 19, tolerance=0.1) is None


def test_recovery_time_immediate_next_sample():
    # The very next sample is already back at level -> recovery 1.
    y = np.concatenate([np.full(10, 1.0), np.full(10, 1.0)])
    assert recovery_time(y, 9, tolerance=0.05) == 1


def test_recovery_time_no_postbreak_data():
    y = np.full(10, 1.0)
    assert recovery_time(y, 9, tolerance=0.1) is None
    assert recovery_time(y, 99, tolerance=0.1) is None


def test_recovery_time_skips_nan_gaps():
    y = np.concatenate([np.full(10, 1.0), [0.3, np.nan, np.nan, 1.0]])
    # Recovery is the 4th post-break sample (NaNs skipped, not counted as
    # recovered, offset still advances).
    assert recovery_time(y, 9, tolerance=0.1) == 4


def test_recovery_time_default_tolerance():
    rng = np.random.default_rng(4)
    pre = 1.0 + rng.normal(0, 0.01, 20)
    post = np.concatenate([np.full(3, 0.3), 1.0 + rng.normal(0, 0.01, 10)])
    y = np.concatenate([pre, post])
    # With the default (pre-break std) tolerance it recovers after the dip.
    rec = recovery_time(y, 19)
    assert rec is not None and rec >= 3
