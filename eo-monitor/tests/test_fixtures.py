"""Verify the committed fixture loads and drives the anomaly math correctly."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eo_monitor.anomaly import baseline_statistics

FIXTURE = Path(__file__).parent / "fixtures" / "baseline_stack.npy"


def test_baseline_fixture_loads() -> None:
    stack = np.load(FIXTURE)
    assert stack.shape == (3, 2, 2)


def test_baseline_fixture_statistics() -> None:
    stack = np.load(FIXTURE)
    mean, std = baseline_statistics(stack)
    # Pixel [0,0]: [0,2,4] -> mean 2, population std = sqrt(8/3).
    # Pixel [0,1]: constant 10 -> mean 10, std 0.
    # Pixel [1,0]: [1,3,5] -> mean 3, std sqrt(8/3).
    # Pixel [1,1]: constant 5 -> mean 5, std 0.
    np.testing.assert_allclose(mean, np.array([[2.0, 10.0], [3.0, 5.0]]))
    np.testing.assert_allclose(
        std,
        np.array([[np.sqrt(8 / 3), 0.0], [np.sqrt(8 / 3), 0.0]]),
    )
