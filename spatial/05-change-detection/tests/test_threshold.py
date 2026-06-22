"""Hand-derived known-answer tests for Otsu thresholding.

A clean two-Gaussian mixture must threshold in the valley between the modes; a
tiny known array must match a hand-derived split; NaN must be ignored. Numpy +
stdlib only.
"""

from __future__ import annotations

import numpy as np
import pytest

from floodmap.threshold import histogram_modes, otsu_threshold


def test_otsu_lands_in_valley_of_bimodal_histogram() -> None:
    """Two well-separated Gaussians (mean 0 and 10) split near the midpoint.

    The valley between the modes is around 5, so the Otsu threshold must fall
    comfortably between the two peaks (3..7).
    """
    rng = np.random.default_rng(0)
    low = rng.normal(0.0, 1.0, size=5000)
    high = rng.normal(10.0, 1.0, size=5000)
    values = np.concatenate([low, high])
    t = otsu_threshold(values, bins=256)
    assert 3.0 < t < 7.0


def test_otsu_known_small_array() -> None:
    """A clearly bimodal 8-value array thresholds between the two clusters.

    Values: four near 1 (0,1,2,1) and four near 100 (100,101,99,100). With the
    range [0, 101] split into 256 bins, the between-class variance is maximised
    by putting the low cluster in class 0 and the high cluster in class 1, so the
    threshold lands strictly between 2 and 99.
    """
    values = np.array([0.0, 1.0, 2.0, 1.0, 100.0, 101.0, 99.0, 100.0])
    t = otsu_threshold(values, bins=256)
    assert 2.0 < t < 99.0


def test_otsu_is_nan_aware() -> None:
    """NaNs are ignored: padding the bimodal array with NaNs keeps the threshold."""
    values = np.array([0.0, 1.0, 2.0, 1.0, 100.0, 101.0, 99.0, 100.0])
    with_nan = np.concatenate([values, [np.nan, np.nan, np.nan]])
    assert otsu_threshold(with_nan, bins=256) == otsu_threshold(values, bins=256)


def test_histogram_modes_recovers_two_peaks() -> None:
    """The low/high mode estimates bracket the Otsu threshold."""
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(-18.0, 0.5, 4000), rng.normal(-7.0, 0.5, 4000)])
    low, high = histogram_modes(values, bins=128)
    t = otsu_threshold(values, bins=128)
    assert low < t < high
    assert low == pytest.approx(-18.0, abs=1.0)
    assert high == pytest.approx(-7.0, abs=1.0)


def test_otsu_invalid_inputs_raise() -> None:
    """Too few bins, all-NaN, or a constant array raise ValueError."""
    with pytest.raises(ValueError):
        otsu_threshold(np.array([1.0, 2.0, 3.0]), bins=1)
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan, np.nan]))
    with pytest.raises(ValueError):
        otsu_threshold(np.array([5.0, 5.0, 5.0]))
