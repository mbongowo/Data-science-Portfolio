"""Hand-checked unit tests for the spectral index math.

All expected values are computed by hand from the normalised-difference
definitions so a reviewer can verify them without running code.
"""

from __future__ import annotations

import numpy as np
import pytest

from eo_monitor.indices import (
    BAND_ALIASES,
    compute_index,
    ndmi,
    ndvi,
    ndwi,
    required_bands,
)


def test_ndvi_known_values() -> None:
    # NDVI = (NIR - Red) / (NIR + Red)
    nir = np.array([0.5, 0.8, 0.2])
    red = np.array([0.1, 0.2, 0.2])
    # (0.5-0.1)/(0.5+0.1) = 0.4/0.6 = 0.666...
    # (0.8-0.2)/(0.8+0.2) = 0.6/1.0 = 0.6
    # (0.2-0.2)/(0.2+0.2) = 0.0/0.4 = 0.0
    expected = np.array([0.4 / 0.6, 0.6, 0.0])
    np.testing.assert_allclose(ndvi(nir, red), expected)


def test_ndwi_known_values() -> None:
    # NDWI = (Green - NIR) / (Green + NIR)
    green = np.array([0.3, 0.4])
    nir = np.array([0.1, 0.6])
    # (0.3-0.1)/(0.3+0.1) = 0.2/0.4 = 0.5
    # (0.4-0.6)/(0.4+0.6) = -0.2/1.0 = -0.2
    expected = np.array([0.5, -0.2])
    np.testing.assert_allclose(ndwi(green, nir), expected)


def test_ndmi_known_values() -> None:
    # NDMI = (NIR - SWIR) / (NIR + SWIR)
    nir = np.array([0.6, 0.5])
    swir = np.array([0.2, 0.5])
    # (0.6-0.2)/(0.6+0.2) = 0.4/0.8 = 0.5
    # (0.5-0.5)/(0.5+0.5) = 0.0/1.0 = 0.0
    expected = np.array([0.5, 0.0])
    np.testing.assert_allclose(ndmi(nir, swir), expected)


def test_divide_by_zero_is_nan() -> None:
    # Both bands zero -> denominator zero -> NaN, not an exception.
    out = ndvi(np.array([0.0]), np.array([0.0]))
    assert np.isnan(out).all()


def test_ndmi_divide_by_zero_is_nan() -> None:
    # NIR and SWIR both zero -> denominator zero -> NaN.
    out = ndmi(np.array([0.5, 0.0]), np.array([0.5, 0.0]))
    # (0.5-0.5)/(0.5+0.5) = 0.0 ; (0.0-0.0)/(0.0+0.0) = NaN.
    assert out[0] == 0.0
    assert np.isnan(out[1])


def test_nan_input_propagates() -> None:
    # A NaN reflectance (e.g. a masked cloud pixel) gives a NaN index, and the
    # other pixels are unaffected.
    green = np.array([0.3, np.nan])
    nir = np.array([0.1, 0.2])
    out = ndwi(green, nir)
    # Pixel 0: (0.3-0.1)/(0.3+0.1) = 0.2/0.4 = 0.5 ; pixel 1: NaN in -> NaN out.
    assert np.isclose(out[0], 0.5)
    assert np.isnan(out[1])


def test_scalar_path_uses_numpy_when_array() -> None:
    # Mixed scalar + array still routes through the numpy branch.
    out = ndvi(np.array([0.8]), 0.2)
    np.testing.assert_allclose(out, np.array([0.6]))


def test_compute_index_dispatch() -> None:
    bands = {
        "nir": np.array([0.8]),
        "red": np.array([0.2]),
        "green": np.array([0.4]),
        "swir": np.array([0.4]),
    }
    np.testing.assert_allclose(compute_index("ndvi", bands), np.array([0.6]))
    np.testing.assert_allclose(compute_index("NDWI", bands), np.array([(0.4 - 0.8) / 1.2]))
    np.testing.assert_allclose(compute_index("NDMI", bands), np.array([(0.8 - 0.4) / 1.2]))


def test_compute_index_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown index"):
        compute_index("EVI", {"nir": np.array([0.1])})


def test_required_bands_maps_to_assets() -> None:
    assert required_bands(["NDVI"]) == sorted([BAND_ALIASES["red"], BAND_ALIASES["nir"]])
    # Union across all three indices = red, green, nir, swir assets.
    bands = required_bands(["NDVI", "NDWI", "NDMI"])
    assert set(bands) == {
        BAND_ALIASES["red"],
        BAND_ALIASES["green"],
        BAND_ALIASES["nir"],
        BAND_ALIASES["swir"],
    }
