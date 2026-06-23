"""Known-answer tests for the US-EPA AQI core.

Expected values are hand-derived from the EPA breakpoint tables (see the module
docstring of :mod:`aqstream.aqi`), so a green test proves the arithmetic, not
merely that it runs. No third-party engine is involved.

Worked values:

* PM2.5 12.0 -> AQI 50  (top of the Good band, I_hi).
* PM2.5 35.4 -> AQI 100 (top of the Moderate band, I_hi).
* PM2.5 9.0  -> (50/12)*9 = 37.5 -> rounds half-up to 38.
* PM2.5 35.5 -> AQI 101 (bottom of the USG band, I_lo).
* PM10  54   -> AQI 50  (top of the Good band).
* PM10  155  -> AQI 101 (bottom of the USG band).
"""

from __future__ import annotations

import pytest

from aqstream import (
    aqi_category,
    aqi_from_pollutants,
    pm10_to_aqi,
    pm25_to_aqi,
)


def test_pm25_breakpoint_boundaries() -> None:
    assert pm25_to_aqi(12.0) == 50
    assert pm25_to_aqi(35.4) == 100
    assert pm25_to_aqi(35.5) == 101


def test_pm25_midrange_rounds_half_up() -> None:
    """(50/12)*9 = 37.5 -> 38 (half away from zero)."""
    assert pm25_to_aqi(9.0) == 38


def test_pm10_known_values() -> None:
    assert pm10_to_aqi(54) == 50
    assert pm10_to_aqi(155) == 101


def test_pm25_truncates_before_lookup() -> None:
    """12.04 truncates to 12.0 -> still the top of the Good band (50)."""
    assert pm25_to_aqi(12.04) == 50


def test_aqi_from_pollutants_takes_the_max() -> None:
    """PM2.5 35.5 -> 101; PM10 50 -> 46; the overall AQI is the larger, 101."""
    assert pm25_to_aqi(35.5) == 101
    assert pm10_to_aqi(50) == 46
    assert aqi_from_pollutants(pm25=35.5, pm10=50) == 101


def test_aqi_from_pollutants_single_pollutant() -> None:
    assert aqi_from_pollutants(pm25=12.0) == 50
    assert aqi_from_pollutants(pm10=54) == 50


def test_aqi_from_pollutants_requires_one() -> None:
    with pytest.raises(ValueError):
        aqi_from_pollutants()


def test_aqi_category_boundaries() -> None:
    assert aqi_category(0) == "Good"
    assert aqi_category(50) == "Good"
    assert aqi_category(51) == "Moderate"
    assert aqi_category(100) == "Moderate"
    assert aqi_category(101) == "Unhealthy for Sensitive Groups"
    assert aqi_category(150) == "Unhealthy for Sensitive Groups"
    assert aqi_category(151) == "Unhealthy"
    assert aqi_category(200) == "Unhealthy"
    assert aqi_category(201) == "Very Unhealthy"
    assert aqi_category(300) == "Very Unhealthy"
    assert aqi_category(301) == "Hazardous"
    assert aqi_category(500) == "Hazardous"


def test_negative_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        pm25_to_aqi(-1.0)
    with pytest.raises(ValueError):
        aqi_category(-1.0)


def test_offscale_concentration_rejected() -> None:
    with pytest.raises(ValueError):
        pm25_to_aqi(10_000.0)
