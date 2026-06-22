"""Hand-checked unit tests for the spectral index math.

All expected values are computed by hand from the normalised-difference
definitions so a reviewer can verify them without running code.
"""

from __future__ import annotations

import numpy as np
import pytest

from eo_monitor.indices import (
    BAND_ALIASES,
    arvi,
    awei,
    bai,
    bsi,
    ci_green,
    ci_rededge,
    clay_minerals,
    compute_index,
    dvi,
    evi,
    evi2,
    ferrous_minerals,
    gndvi,
    ibi,
    iron_oxide,
    lai,
    mcari,
    mndwi,
    msavi,
    nbr,
    nbr2,
    ndbi,
    ndgi,
    ndii,
    ndmi,
    ndre,
    ndsi,
    ndvi,
    ndwi,
    normalized_difference,
    required_bands,
    rvi,
    salinity_index,
    savi,
    tcari,
    ui,
    vari,
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
        compute_index("NOSUCHINDEX", {"nir": np.array([0.1])})


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


# --------------------------------------------------------------------------- #
# Full catalogue: hand-derived known-answer tests
# --------------------------------------------------------------------------- #
# Scale-dependent indices (EVI, MSAVI, AWEI, BAI, LAI) are tested with
# reflectance in [0, 1] as documented in indices.py.


def test_evi_known_values() -> None:
    # EVI = 2.5*(NIR-Red)/(NIR + 6*Red - 7.5*Blue + 1).
    nir = np.array([0.5])
    red = np.array([0.1])
    blue = np.array([0.02])
    # 2.5*(0.4) / (0.5 + 0.6 - 0.15 + 1) = 1.0 / 1.95
    np.testing.assert_allclose(evi(nir, red, blue), np.array([1.0 / 1.95]))


def test_msavi_known_values() -> None:
    # MSAVI = (2*NIR+1 - sqrt((2*NIR+1)^2 - 8*(NIR-Red)))/2.
    nir = np.array([0.5])
    red = np.array([0.1])
    # 2*0.5+1 = 2.0 ; 2.0^2 - 8*0.4 = 4 - 3.2 = 0.8 ; sqrt(0.8)
    expected = (2.0 - np.sqrt(0.8)) / 2.0
    np.testing.assert_allclose(msavi(nir, red), np.array([expected]))


def test_gndvi_known_values() -> None:
    # (NIR-Green)/(NIR+Green) = (0.6-0.2)/(0.6+0.2) = 0.5
    np.testing.assert_allclose(gndvi(np.array([0.6]), np.array([0.2])), np.array([0.5]))


def test_arvi_known_values() -> None:
    # rb = 2*Red - Blue = 2*0.2 - 0.1 = 0.3 ; (NIR-rb)/(NIR+rb) = (0.7-0.3)/(0.7+0.3)=0.4
    out = arvi(np.array([0.7]), np.array([0.2]), np.array([0.1]))
    np.testing.assert_allclose(out, np.array([0.4]))


def test_ndre_known_values() -> None:
    # (NIR-RE)/(NIR+RE) = (0.6-0.3)/(0.6+0.3) = 0.3/0.9 = 1/3
    np.testing.assert_allclose(ndre(np.array([0.6]), np.array([0.3])), np.array([1.0 / 3.0]))


def test_vari_known_values() -> None:
    # (Green-Red)/(Green+Red-Blue) = (0.3-0.2)/(0.3+0.2-0.1) = 0.1/0.4 = 0.25
    out = vari(np.array([0.3]), np.array([0.2]), np.array([0.1]))
    np.testing.assert_allclose(out, np.array([0.25]))


def test_rvi_dvi_known_values() -> None:
    np.testing.assert_allclose(rvi(np.array([0.6]), np.array([0.2])), np.array([3.0]))
    np.testing.assert_allclose(dvi(np.array([0.6]), np.array([0.2])), np.array([0.4]))


def test_ci_known_values() -> None:
    # NIR/Green - 1 = 0.6/0.2 - 1 = 2.0 ; NIR/RE - 1 = 0.6/0.3 - 1 = 1.0
    np.testing.assert_allclose(ci_green(np.array([0.6]), np.array([0.2])), np.array([2.0]))
    np.testing.assert_allclose(ci_rededge(np.array([0.6]), np.array([0.3])), np.array([1.0]))


def test_mcari_known_values() -> None:
    # ((RE-Red) - 0.2*(RE-Green)) * (RE/Red)
    re, red, green = np.array([0.4]), np.array([0.2]), np.array([0.3])
    # ((0.2) - 0.2*(0.1)) * (0.4/0.2) = (0.2 - 0.02) * 2 = 0.18*2 = 0.36
    np.testing.assert_allclose(mcari(re, red, green), np.array([0.36]))


def test_tcari_known_values() -> None:
    # 3*((RE-Red) - 0.2*(RE-Green)*(RE/Red))
    re, red, green = np.array([0.4]), np.array([0.2]), np.array([0.3])
    # 3*((0.2) - 0.2*(0.1)*(2.0)) = 3*(0.2 - 0.04) = 3*0.16 = 0.48
    np.testing.assert_allclose(tcari(re, red, green), np.array([0.48]))


def test_lai_matches_evi_relation() -> None:
    nir, red, blue = np.array([0.5]), np.array([0.1]), np.array([0.02])
    np.testing.assert_allclose(lai(nir, red, blue), 3.618 * evi(nir, red, blue) - 0.118)


def test_mndwi_known_values() -> None:
    # (Green-SWIR1)/(Green+SWIR1) = (0.4-0.1)/(0.4+0.1) = 0.3/0.5 = 0.6
    np.testing.assert_allclose(mndwi(np.array([0.4]), np.array([0.1])), np.array([0.6]))


def test_awei_known_values() -> None:
    # 4*(Green-SWIR1) - (0.25*NIR + 2.75*SWIR2)
    g, n, s1, s2 = np.array([0.3]), np.array([0.2]), np.array([0.1]), np.array([0.05])
    # 4*(0.2) - (0.05 + 0.1375) = 0.8 - 0.1875 = 0.6125
    np.testing.assert_allclose(awei(g, n, s1, s2), np.array([0.6125]))


def test_ndii_equals_ndmi() -> None:
    nir, swir = np.array([0.6, 0.3]), np.array([0.2, 0.4])
    np.testing.assert_allclose(ndii(nir, swir), ndmi(nir, swir))


def test_bsi_known_values() -> None:
    # ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))
    s1, red, nir, blue = np.array([0.3]), np.array([0.2]), np.array([0.4]), np.array([0.1])
    # ((0.5)-(0.5)) / ((0.5)+(0.5)) = 0/1 = 0
    np.testing.assert_allclose(bsi(s1, red, nir, blue), np.array([0.0]))


def test_salinity_index_known_values() -> None:
    # sqrt(Green*Red) = sqrt(0.4*0.1) = sqrt(0.04) = 0.2
    np.testing.assert_allclose(salinity_index(np.array([0.4]), np.array([0.1])), np.array([0.2]))


def test_geology_ratios_known_values() -> None:
    np.testing.assert_allclose(iron_oxide(np.array([0.4]), np.array([0.1])), np.array([4.0]))
    np.testing.assert_allclose(clay_minerals(np.array([0.3]), np.array([0.15])), np.array([2.0]))
    np.testing.assert_allclose(ferrous_minerals(np.array([0.2]), np.array([0.5])), np.array([0.4]))


def test_ndbi_ui_known_values() -> None:
    # NDBI = (SWIR1-NIR)/(SWIR1+NIR) = (0.3-0.1)/(0.3+0.1) = 0.5
    np.testing.assert_allclose(ndbi(np.array([0.3]), np.array([0.1])), np.array([0.5]))
    # UI = (SWIR2-NIR)/(SWIR2+NIR) = (0.4-0.2)/(0.4+0.2) = 1/3
    np.testing.assert_allclose(ui(np.array([0.4]), np.array([0.2])), np.array([1.0 / 3.0]))


def test_ibi_matches_subindices() -> None:
    s1, nir, red, green = np.array([0.3]), np.array([0.4]), np.array([0.2]), np.array([0.35])
    ndbi_v = ndbi(s1, nir)
    savi_v = savi(nir, red)
    mndwi_v = mndwi(green, s1)
    half = (savi_v + mndwi_v) / 2.0
    expected = (ndbi_v - half) / (ndbi_v + half)
    np.testing.assert_allclose(ibi(s1, nir, red, green), expected)


def test_ndsi_ndgi_known_values() -> None:
    # NDSI = (Green-SWIR1)/(Green+SWIR1) = (0.5-0.1)/(0.5+0.1) = 2/3
    np.testing.assert_allclose(ndsi(np.array([0.5]), np.array([0.1])), np.array([2.0 / 3.0]))
    # NDGI = (Green-Red)/(Green+Red) = (0.4-0.2)/(0.4+0.2) = 1/3
    np.testing.assert_allclose(ndgi(np.array([0.4]), np.array([0.2])), np.array([1.0 / 3.0]))


def test_nbr2_known_values() -> None:
    # (SWIR1-SWIR2)/(SWIR1+SWIR2) = (0.4-0.2)/(0.4+0.2) = 1/3
    np.testing.assert_allclose(nbr2(np.array([0.4]), np.array([0.2])), np.array([1.0 / 3.0]))


def test_bai_known_values() -> None:
    # BAI = 1/((0.1-Red)^2 + (0.06-NIR)^2)
    red, nir = np.array([0.2]), np.array([0.46])
    # (0.1-0.2)^2 + (0.06-0.46)^2 = 0.01 + 0.16 = 0.17 ; 1/0.17
    np.testing.assert_allclose(bai(red, nir), np.array([1.0 / 0.17]))


def test_new_indices_divide_by_zero_is_nan() -> None:
    # RVI: Red == 0 -> NaN.
    assert np.isnan(rvi(np.array([0.5]), np.array([0.0]))).all()
    # iron_oxide: Blue == 0 -> NaN.
    assert np.isnan(iron_oxide(np.array([0.5]), np.array([0.0]))).all()
    # NDBI: both zero -> NaN.
    assert np.isnan(ndbi(np.array([0.0]), np.array([0.0]))).all()
    # BAI: pixel exactly at reference point -> denominator 0 -> NaN.
    assert np.isnan(bai(np.array([0.1]), np.array([0.06]))).all()
    # MSAVI: radicand = (2*NIR-1)^2 + 8*Red < 0 forces NaN. With NIR=0.5 the
    # first term is 0, so any Red < 0 makes it negative: Red=-0.1 -> radicand -0.8.
    assert np.isnan(msavi(np.array([0.5]), np.array([-0.1]))).all()


def test_compute_index_full_catalogue() -> None:
    bands = {
        "nir": np.array([0.5]),
        "red": np.array([0.1]),
        "green": np.array([0.3]),
        "blue": np.array([0.05]),
        "rededge": np.array([0.25]),
        "swir": np.array([0.2]),
        "swir2": np.array([0.15]),
    }
    # Spot-check a few across categories dispatch to the right function.
    np.testing.assert_allclose(
        compute_index("EVI", bands), evi(bands["nir"], bands["red"], bands["blue"])
    )
    np.testing.assert_allclose(compute_index("mndwi", bands), mndwi(bands["green"], bands["swir"]))
    np.testing.assert_allclose(
        compute_index("BSI", bands), bsi(bands["swir"], bands["red"], bands["nir"], bands["blue"])
    )
    np.testing.assert_allclose(compute_index("NBR2", bands), nbr2(bands["swir"], bands["swir2"]))
    np.testing.assert_allclose(compute_index("UI", bands), ui(bands["swir2"], bands["nir"]))


def test_required_bands_union_full() -> None:
    # AWEI needs green, nir, swir1, swir2 -> those four assets.
    assert set(required_bands(["AWEI"])) == {
        BAND_ALIASES["green"],
        BAND_ALIASES["nir"],
        BAND_ALIASES["swir"],
        BAND_ALIASES["swir2"],
    }
    # NDRE needs the red-edge asset.
    assert required_bands(["NDRE"]) == sorted([BAND_ALIASES["nir"], BAND_ALIASES["rededge"]])
