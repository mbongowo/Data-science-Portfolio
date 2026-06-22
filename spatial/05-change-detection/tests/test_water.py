"""Hand-derived known-answer tests for water masking and unit conversion.

Water-mask polarity on a known array; MNDWI of known bands and 0/0 -> NaN; to_db
of known values (10*log10(0.1) = -10 dB) and <=0 -> NaN. Numpy + stdlib only.
"""

from __future__ import annotations

import numpy as np
import pytest

from floodmap.water import mndwi, to_db, water_mask


def test_water_mask_below_polarity_for_sar() -> None:
    """polarity='below' marks pixels below the threshold (dark = water in SAR)."""
    img = np.array([[-20.0, -15.0, -5.0, np.nan]])
    mask = water_mask(img, threshold=-12.0, polarity="below")
    # -20 and -15 are below -12 -> water; -5 is land; NaN is not water.
    assert list(mask[0]) == [True, True, False, False]


def test_water_mask_above_polarity_for_optical() -> None:
    """polarity='above' marks pixels above the threshold (bright = water in MNDWI)."""
    img = np.array([[0.6, 0.1, -0.3, np.nan]])
    mask = water_mask(img, threshold=0.0, polarity="above")
    # 0.6 above 0 -> water; 0.1 above -> water; -0.3 below -> land; NaN -> False.
    assert list(mask[0]) == [True, True, False, False]


def test_water_mask_bad_polarity_raises() -> None:
    """An unknown polarity raises ValueError."""
    with pytest.raises(ValueError):
        water_mask(np.zeros((2, 2)), threshold=0.0, polarity="sideways")


def test_mndwi_known_values_and_div_by_zero() -> None:
    """MNDWI = (G - SWIR)/(G + SWIR); 0/0 -> NaN.

    (0.3, 0.1) -> 0.2/0.4 = 0.5 ; (0.1, 0.1) -> 0/0.2 = 0 ; (0,0) -> NaN.
    """
    green = np.array([[0.3, 0.1, 0.0]])
    swir = np.array([[0.1, 0.1, 0.0]])
    out = mndwi(green, swir)
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(0.0)
    assert np.isnan(out[0, 2])


def test_to_db_known_value_and_nonpositive() -> None:
    """10*log10(0.1) = -10 dB, 10*log10(1) = 0 dB; <=0 -> NaN."""
    linear = np.array([[0.1, 1.0, 0.0, -0.5]])
    db = to_db(linear)
    assert db[0, 0] == pytest.approx(-10.0)
    assert db[0, 1] == pytest.approx(0.0)
    assert np.isnan(db[0, 2])
    assert np.isnan(db[0, 3])
