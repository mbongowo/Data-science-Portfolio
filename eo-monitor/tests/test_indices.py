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
    evi2,
    nbr,
    ndmi,
    ndvi,
    ndwi,
    normalized_difference,
    required_bands,
    savi,
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


def test_normalized_difference_public() -> None:
    # normalized_difference(a, b) = (a - b) / (a + b).
    a = np.array([0.8, 0.5])
    b = np.array([0.2, 0.5])
    # (0.8-0.2)/(0.8+0.2) = 0.6 ; (0.5-0.5)/(0.5+0.5) = 0.0
    np.testing.assert_allclose(normalized_difference(a, b), np.array([0.6, 0.0]))
    # And it matches ndvi (same formula).
    np.testing.assert_allclose(normalized_difference(a, b), ndvi(a, b))


def test_savi_known_values() -> None:
    # SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L), default L = 0.5.
    nir = np.array([0.5, 0.45])
    red = np.array([0.1, 0.06])
    # Pixel 0: 1.5 * (0.5-0.1) / (0.5+0.1+0.5) = 1.5*0.4/1.1 = 0.6/1.1 = 0.545454...
    # Pixel 1: 1.5 * (0.45-0.06) / (0.45+0.06+0.5) = 1.5*0.39/1.01 = 0.585/1.01
    expected = np.array([0.6 / 1.1, 0.585 / 1.01])
    np.testing.assert_allclose(savi(nir, red), expected)


def test_savi_L_zero_equals_ndvi() -> None:
    # At L = 0, SAVI collapses to NDVI.
    nir = np.array([0.6, 0.3])
    red = np.array([0.2, 0.3])
    np.testing.assert_allclose(savi(nir, red, L=0.0), ndvi(nir, red))


def test_evi2_known_values() -> None:
    # EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1).
    nir = np.array([0.5, 0.3])
    red = np.array([0.1, 0.3])
    # Pixel 0: 2.5*(0.5-0.1)/(0.5+2.4*0.1+1) = 2.5*0.4/1.74 = 1.0/1.74 = 0.574712...
    # Pixel 1: 2.5*(0.3-0.3)/(...) = 0.0
    expected = np.array([1.0 / 1.74, 0.0])
    np.testing.assert_allclose(evi2(nir, red), expected)


def test_nbr_known_values() -> None:
    # NBR = (NIR - SWIR) / (NIR + SWIR).
    nir = np.array([0.6, 0.4])
    swir2 = np.array([0.2, 0.4])
    # (0.6-0.2)/(0.6+0.2) = 0.5 ; (0.4-0.4)/(0.4+0.4) = 0.0
    np.testing.assert_allclose(nbr(nir, swir2), np.array([0.5, 0.0]))


def test_savi_divide_by_zero_is_nan() -> None:
    # NIR + Red + L == 0 only if NIR = Red = -L; force it with L = 0 and zeros.
    out = savi(np.array([0.0]), np.array([0.0]), L=0.0)
    assert np.isnan(out).all()


def test_evi2_nan_input_propagates() -> None:
    out = evi2(np.array([0.5, np.nan]), np.array([0.1, 0.1]))
    assert np.isclose(out[0], 1.0 / 1.74)
    assert np.isnan(out[1])


def test_compute_index_new_names() -> None:
    bands = {
        "nir": np.array([0.5]),
        "red": np.array([0.1]),
        "swir2": np.array([0.2]),
    }
    np.testing.assert_allclose(compute_index("SAVI", bands), savi(bands["nir"], bands["red"]))
    np.testing.assert_allclose(compute_index("evi2", bands), evi2(bands["nir"], bands["red"]))
    np.testing.assert_allclose(compute_index("NBR", bands), nbr(bands["nir"], bands["swir2"]))


def test_required_bands_new_indices() -> None:
    assert required_bands(["NBR"]) == sorted([BAND_ALIASES["nir"], BAND_ALIASES["swir2"]])
    assert set(required_bands(["SAVI", "EVI2"])) == {BAND_ALIASES["red"], BAND_ALIASES["nir"]}
