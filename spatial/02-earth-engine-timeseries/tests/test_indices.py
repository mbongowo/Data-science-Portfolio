"""Hand-derived known-answer tests for the pure-numpy spectral indices.

Every expected value is worked out by hand so a green test proves the formula is
correct, not merely that it runs. None of these tests import the geospatial or
Earth Engine stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from eets.indices import nbr, ndvi, ndwi, normalized_difference


def test_normalized_difference_known_value() -> None:
    """(3 - 1) / (3 + 1) = 0.5; (1 - 3) / (1 + 3) = -0.5."""
    assert normalized_difference(np.array([3.0]), np.array([1.0]))[0] == pytest.approx(
        0.5
    )
    assert normalized_difference(np.array([1.0]), np.array([3.0]))[0] == pytest.approx(
        -0.5
    )


def test_ndvi_known_bands() -> None:
    """NDVI of NIR=0.4, Red=0.1 is (0.4-0.1)/(0.4+0.1) = 0.3/0.5 = 0.6."""
    val = ndvi(np.array([0.4]), np.array([0.1]))[0]
    assert val == pytest.approx(0.6)


def test_ndwi_known_bands() -> None:
    """NDWI of Green=0.3, NIR=0.1 is (0.3-0.1)/(0.3+0.1) = 0.2/0.4 = 0.5."""
    val = ndwi(np.array([0.3]), np.array([0.1]))[0]
    assert val == pytest.approx(0.5)


def test_nbr_known_bands() -> None:
    """NBR of NIR=0.5, SWIR=0.1 is (0.5-0.1)/(0.5+0.1) = 0.4/0.6 = 2/3."""
    val = nbr(np.array([0.5]), np.array([0.1]))[0]
    assert val == pytest.approx(2.0 / 3.0)


def test_div_by_zero_is_nan_not_inf() -> None:
    """Both bands zero -> denominator zero -> NaN, never inf or 0."""
    out = normalized_difference(np.array([0.0, 0.4]), np.array([0.0, 0.1]))
    assert np.isnan(out[0])
    assert np.isfinite(out[1])
    assert out[1] == pytest.approx(0.6)


def test_indices_are_elementwise_over_arrays() -> None:
    """NDVI is computed element-wise across a 2-D array."""
    nir = np.array([[0.4, 0.0], [0.6, 0.2]])
    red = np.array([[0.1, 0.0], [0.2, 0.2]])
    out = ndvi(nir, red)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(0.6)
    assert np.isnan(out[0, 1])  # 0/0
    assert out[1, 0] == pytest.approx(0.5)  # (0.6-0.2)/0.8
    assert out[1, 1] == pytest.approx(0.0)  # (0.2-0.2)/0.4
