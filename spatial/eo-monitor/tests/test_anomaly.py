"""Hand-checked unit tests for the z-score anomaly math."""

from __future__ import annotations

import numpy as np

from eo_monitor.anomaly import (
    anomaly_cube,
    anomaly_fraction,
    baseline_statistics,
    classify_anomaly,
    robust_zscore,
    zscore_anomaly,
)
from eo_monitor.indices import ndvi


def test_baseline_statistics_numpy() -> None:
    # Baseline stack of 4 time slices over a 2-pixel grid.
    # Pixel A: [0, 2, 4, 6] -> mean 3, population std = sqrt(5) = 2.2360679...
    # Pixel B: [10, 10, 10, 10] -> mean 10, std 0.
    baseline = np.array(
        [
            [0.0, 10.0],
            [2.0, 10.0],
            [4.0, 10.0],
            [6.0, 10.0],
        ]
    )
    mean, std = baseline_statistics(baseline)
    np.testing.assert_allclose(mean, np.array([3.0, 10.0]))
    np.testing.assert_allclose(std, np.array([np.sqrt(5.0), 0.0]))


def test_zscore_anomaly_known_values() -> None:
    # z = (value - mean) / std
    value = np.array([5.0, 1.0])
    mean = np.array([3.0, 1.0])
    std = np.array([2.0, 4.0])
    # (5-3)/2 = 1.0 ; (1-1)/4 = 0.0
    np.testing.assert_allclose(zscore_anomaly(value, mean, std), np.array([1.0, 0.0]))


def test_zscore_zero_std_is_nan() -> None:
    out = zscore_anomaly(np.array([5.0]), np.array([3.0]), np.array([0.0]))
    assert np.isnan(out).all()


def test_anomaly_cube_end_to_end() -> None:
    # Baseline pixel A: [0,2,4,6] -> mean 3, std sqrt(5).
    # Baseline pixel B: constant 10 -> std 0 -> anomaly NaN.
    baseline = np.array(
        [
            [0.0, 10.0],
            [2.0, 10.0],
            [4.0, 10.0],
            [6.0, 10.0],
        ]
    )
    # Single target slice.
    target = np.array([3.0 + np.sqrt(5.0), 12.0])
    z = anomaly_cube(target, baseline)
    # Pixel A: ((3+sqrt5) - 3)/sqrt5 = 1.0 ; Pixel B: std 0 -> NaN.
    assert np.isclose(z[0], 1.0)
    assert np.isnan(z[1])


def test_anomaly_cube_handles_nan_in_baseline() -> None:
    # NaNs (e.g. masked clouds) are ignored in baseline stats.
    baseline = np.array(
        [
            [0.0],
            [np.nan],
            [4.0],
        ]
    )
    # Valid values [0, 4] -> mean 2, population std 2.
    mean, std = baseline_statistics(baseline)
    np.testing.assert_allclose(mean, np.array([2.0]))
    np.testing.assert_allclose(std, np.array([2.0]))
    z = anomaly_cube(np.array([4.0]), baseline)
    np.testing.assert_allclose(z, np.array([1.0]))


def test_indices_then_anomaly_chained() -> None:
    # Drive the real path: compute NDVI on a tiny synthetic stack, then take the
    # z-score anomaly of a target NDVI map against that baseline. Grid is
    # (time=3, y=1, x=2). Band values are chosen so every NDVI is exact.
    nir_base = np.array([[[0.6, 0.5]], [[0.6, 0.5]], [[0.9, 0.5]]])
    red_base = np.array([[[0.4, 0.5]], [[0.2, 0.5]], [[0.1, 0.5]]])
    ndvi_base = ndvi(nir_base, red_base)
    # Pixel (0,0) NDVI = [0.2, 0.5, 0.8] -> mean 0.5, population std sqrt(0.06).
    # Pixel (0,1) NDVI = [0.0, 0.0, 0.0] -> std 0 -> anomaly NaN.
    mean, std = baseline_statistics(ndvi_base)
    np.testing.assert_allclose(mean, np.array([[0.5, 0.0]]))
    np.testing.assert_allclose(std, np.array([[np.sqrt(0.06), 0.0]]))

    # Target NDVI: pixel (0,0) = 0.5 (equal to baseline mean -> z 0),
    # pixel (0,1) = 0.2 but baseline std 0 -> NaN.
    target = ndvi(np.array([[0.6, 0.6]]), np.array([[0.2, 0.4]]))
    np.testing.assert_allclose(target, np.array([[0.5, 0.2]]))

    z = anomaly_cube(target, ndvi_base)
    assert np.isclose(z[0, 0], 0.0, atol=1e-12)
    assert np.isnan(z[0, 1])


def test_robust_zscore_known_values() -> None:
    # Baseline pixel A: [1, 2, 4, 100] -> median 3,
    #   |x - 3| = [2, 1, 1, 97] -> median 1.5, MAD = 1.4826 * 1.5 = 2.2239.
    # Baseline pixel B: constant 5 -> MAD 0 -> z NaN.
    baseline = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [4.0, 5.0],
            [100.0, 5.0],
        ]
    )
    mad_a = 1.4826 * 1.5
    # value 3 -> z 0 ; value 3 + mad_a -> z 1 for pixel A.
    value = np.array([3.0 + mad_a, 7.0])
    z = robust_zscore(value, baseline)
    assert np.isclose(z[0], 1.0)
    assert np.isnan(z[1])


def test_robust_zscore_resists_outlier() -> None:
    # A single huge outlier barely moves the median/MAD, unlike mean/std.
    baseline = np.array([[1.0], [2.0], [3.0], [4.0], [1000.0]])
    # median = 3, |x-3| = [2,1,0,1,997] -> median 1 -> MAD = 1.4826.
    z = robust_zscore(np.array([3.0 + 1.4826]), baseline)
    np.testing.assert_allclose(z, np.array([1.0]))


def test_robust_zscore_zero_mad_is_nan() -> None:
    baseline = np.array([[7.0], [7.0], [7.0]])
    z = robust_zscore(np.array([9.0]), baseline)
    assert np.isnan(z).all()


def test_robust_zscore_ignores_nan() -> None:
    # NaN baseline observation is skipped in median/MAD.
    baseline = np.array([[1.0], [np.nan], [3.0], [5.0]])
    # finite [1,3,5] -> median 3, |x-3| = [2,0,2] -> median 2 -> MAD = 2.9652.
    z = robust_zscore(np.array([3.0 + 2.9652]), baseline)
    np.testing.assert_allclose(z, np.array([1.0]), atol=1e-4)


def test_anomaly_fraction_known() -> None:
    # 5 finite pixels, 2 exceed |z| > 2 (-3 and 4); one NaN excluded.
    z = np.array([0.5, -3.0, 1.0, 4.0, np.nan, 2.0])
    # |z| > 2 strictly: -3 and 4 qualify; 2.0 does not (not strictly > 2).
    assert anomaly_fraction(z, threshold=2.0) == 2 / 5


def test_anomaly_fraction_all_nan_is_zero() -> None:
    z = np.array([np.nan, np.nan])
    assert anomaly_fraction(z) == 0.0


def test_classify_anomaly_known() -> None:
    z = np.array([-3.0, -1.0, 0.0, 1.5, 2.5, np.nan])
    # < -2 -> -1 ; within [-2, 2] -> 0 ; > 2 -> +1 ; NaN -> 0.
    expected = np.array([-1, 0, 0, 0, 1, 0], dtype="int8")
    np.testing.assert_array_equal(classify_anomaly(z, threshold=2.0), expected)
    assert classify_anomaly(z).dtype == np.int8


def test_classify_anomaly_empty() -> None:
    out = classify_anomaly(np.array([]))
    assert out.shape == (0,)
    assert out.dtype == np.int8
