"""Hand-checked unit tests for the z-score anomaly math."""

from __future__ import annotations

import numpy as np

from eo_monitor.anomaly import anomaly_cube, baseline_statistics, zscore_anomaly
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
