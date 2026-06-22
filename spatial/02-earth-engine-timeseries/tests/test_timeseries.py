"""Hand-derived known-answer tests for the time-series + compositing core.

Tiny stacks with hand-computed reductions. NaN handling is checked explicitly:
masked / cloudy pixels are skipped, never read as 0. Numpy + stdlib only.
"""

from __future__ import annotations

import numpy as np
import pytest

from eets.timeseries import index_timeseries, mask_invalid, temporal_composite


def test_index_timeseries_spatial_means() -> None:
    """Per-time-step spatial mean of a 2-scene 2x2 stack.

    Scene 0 = [[1, 2], [3, 4]] -> mean 2.5.
    Scene 1 = [[0, 0], [2, 2]] -> mean 1.0.
    """
    stack = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [2.0, 2.0]],
        ]
    )
    series = index_timeseries(stack, axis=0)
    assert series.shape == (2,)
    assert series[0] == pytest.approx(2.5)
    assert series[1] == pytest.approx(1.0)


def test_index_timeseries_is_nan_aware() -> None:
    """A NaN pixel is skipped: mean of [2, NaN, 4] is 3.0, not NaN."""
    stack = np.array([[[2.0, np.nan], [4.0, 6.0]]])  # one scene, mean of {2,4,6}=4
    series = index_timeseries(stack, axis=0)
    assert series[0] == pytest.approx(4.0)


def test_temporal_composite_median_and_mean() -> None:
    """Per-pixel median and mean over time on a 3-scene stack.

    Pixel (0,0) values over time = [1, 2, 9]: median 2, mean 4.
    Pixel (0,1) values over time = [4, 4, 4]: median 4, mean 4.
    """
    stack = np.array(
        [
            [[1.0, 4.0]],
            [[2.0, 4.0]],
            [[9.0, 4.0]],
        ]
    )
    med = temporal_composite(stack, agg="median", axis=0)
    mean = temporal_composite(stack, agg="mean", axis=0)
    assert med[0, 0] == pytest.approx(2.0)
    assert med[0, 1] == pytest.approx(4.0)
    assert mean[0, 0] == pytest.approx(4.0)
    assert mean[0, 1] == pytest.approx(4.0)


def test_temporal_composite_skips_nan() -> None:
    """Median over [3, NaN, 5] is 4.0 (NaN dropped)."""
    stack = np.array([[[3.0]], [[np.nan]], [[5.0]]])
    med = temporal_composite(stack, agg="median", axis=0)
    assert med[0, 0] == pytest.approx(4.0)


def test_mask_invalid_sets_the_right_pixels_to_nan() -> None:
    """Pixels whose SCL is in the invalid set become NaN; others untouched.

    SCL = [[4, 9], [3, 4]]; invalid = (3, 9). So (0,1) and (1,0) -> NaN,
    (0,0) and (1,1) keep their band values.
    """
    band = np.array([[0.8, 0.7], [0.6, 0.5]])
    scl = np.array([[4, 9], [3, 4]])
    out = mask_invalid(band, scl, (3, 9))
    assert out[0, 0] == pytest.approx(0.8)
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 0])
    assert out[1, 1] == pytest.approx(0.5)
    # original is not mutated
    assert band[0, 1] == pytest.approx(0.7)


def test_invalid_inputs_raise() -> None:
    """Bad shapes / aggregations raise ValueError."""
    with pytest.raises(ValueError):
        index_timeseries(np.array([1.0, 2.0]))  # 1-D, no spatial extent
    with pytest.raises(ValueError):
        temporal_composite(np.array([1.0, 2.0]))  # 1-D
    with pytest.raises(ValueError):
        temporal_composite(np.zeros((2, 2, 2)), agg="sum")  # bad agg
    with pytest.raises(ValueError):
        mask_invalid(np.zeros((2, 2)), np.zeros((3, 3)), (1,))  # shape mismatch
