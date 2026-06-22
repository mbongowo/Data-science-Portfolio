"""Hand-derived known-answer tests for change detection and burn severity.

Tiny arrays with hand-computed deltas, classifications, and hectares. The
burn-severity test places values exactly on and just inside the USGS dNBR break
points. Numpy + stdlib only.
"""

from __future__ import annotations

import numpy as np
import pytest

from eets.change import (
    change_map,
    change_stats,
    classify_burn_severity,
    classify_change,
    dnbr,
    severity_stats,
)


def test_change_map_delta_on_known_arrays() -> None:
    """change_map is after - before, element-wise; NaN propagates."""
    before = np.array([[0.8, 0.5], [0.2, np.nan]])
    after = np.array([[0.6, 0.9], [0.2, 0.4]])
    delta = change_map(before, after)
    assert delta[0, 0] == pytest.approx(-0.2)  # loss
    assert delta[0, 1] == pytest.approx(0.4)  # gain
    assert delta[1, 0] == pytest.approx(0.0)  # stable
    assert np.isnan(delta[1, 1])  # NaN propagates


def test_classify_change_thresholds() -> None:
    """Loss at/below -0.2, gain at/above +0.2, else stable; NaN -> 0.

    delta = [-0.5, -0.2, -0.1, 0.0, 0.2, 0.5, NaN] with loss=-0.2, gain=0.2
    -> [-1, -1, 0, 0, 1, 1, 0].
    """
    delta = np.array([[-0.5, -0.2, -0.1, 0.0, 0.2, 0.5, np.nan]])
    cls = classify_change(delta, loss_thresh=-0.2, gain_thresh=0.2)
    assert list(cls[0]) == [-1, -1, 0, 0, 1, 1, 0]


def test_change_stats_hectares_from_known_classified() -> None:
    """Counts and hectares from a known classified array at 10 m pixels.

    4 loss pixels and 2 gain pixels on a 10x10 m grid: each pixel is 100 m2 =
    0.01 ha. So loss = 0.04 ha, gain = 0.02 ha. Total 10 pixels here ->
    loss_fraction 0.4, gain_fraction 0.2.
    """
    classified = np.array(
        [
            [-1, -1, -1, -1, 1],
            [1, 0, 0, 0, 0],
        ]
    )
    stats = change_stats(classified, pixel_size_m=10.0)
    assert stats["loss_pixels"] == 4
    assert stats["gain_pixels"] == 2
    assert stats["loss_hectares"] == pytest.approx(0.04)
    assert stats["gain_hectares"] == pytest.approx(0.02)
    assert stats["loss_fraction"] == pytest.approx(0.4)
    assert stats["gain_fraction"] == pytest.approx(0.2)


def test_dnbr_known_value() -> None:
    """dNBR is pre - post; positive means burn."""
    pre = np.array([[0.6, 0.3]])
    post = np.array([[0.1, 0.4]])
    d = dnbr(pre, post)
    assert d[0, 0] == pytest.approx(0.5)  # burned
    assert d[0, 1] == pytest.approx(-0.1)  # regrowth / no burn


def test_classify_burn_severity_straddles_usgs_thresholds() -> None:
    """Values on / around the USGS dNBR break points map to the right classes.

    Break points: 0.10, 0.27, 0.44, 0.66. Test values and expected classes:
      0.05 -> 0 (unburned)
      0.10 -> 1 (low, inclusive lower bound)
      0.20 -> 1 (low)
      0.27 -> 2 (moderate-low, inclusive)
      0.44 -> 3 (moderate-high, inclusive)
      0.66 -> 4 (high, inclusive)
      0.90 -> 4 (high)
      NaN  -> 0 (nodata)
    """
    d = np.array([[0.05, 0.10, 0.20, 0.27, 0.44, 0.66, 0.90, np.nan]])
    cls = classify_burn_severity(d)
    assert list(cls[0]) == [0, 1, 1, 2, 3, 4, 4, 0]


def test_severity_stats_hectares_per_class() -> None:
    """Hectares per severity class at 20 m pixels (each = 400 m2 = 0.04 ha).

    classes = [0, 1, 1, 4] -> unburned 1 px (0.04 ha), low 2 px (0.08 ha),
    high 1 px (0.04 ha), the rest 0.
    """
    classes = np.array([[0, 1, 1, 4]])
    stats = severity_stats(classes, pixel_size_m=20.0)
    assert stats["unburned"] == pytest.approx(0.04)
    assert stats["low"] == pytest.approx(0.08)
    assert stats["moderate_low"] == pytest.approx(0.0)
    assert stats["moderate_high"] == pytest.approx(0.0)
    assert stats["high"] == pytest.approx(0.04)


def test_invalid_inputs_raise() -> None:
    """Shape mismatch, non-2D, bad thresholds, bad pixel size raise ValueError."""
    with pytest.raises(ValueError):
        change_map(np.zeros((2, 2)), np.zeros((3, 3)))  # shape mismatch
    with pytest.raises(ValueError):
        change_map(np.zeros(3), np.zeros(3))  # not 2-D
    with pytest.raises(ValueError):
        classify_change(np.zeros((2, 2)), loss_thresh=0.3, gain_thresh=0.1)  # loss>gain
    with pytest.raises(ValueError):
        change_stats(np.zeros((2, 2)), pixel_size_m=0.0)  # bad pixel size
    with pytest.raises(ValueError):
        classify_burn_severity(np.zeros(3))  # not 2-D
    with pytest.raises(ValueError):
        severity_stats(np.zeros((2, 2)), pixel_size_m=-1.0)  # bad pixel size
