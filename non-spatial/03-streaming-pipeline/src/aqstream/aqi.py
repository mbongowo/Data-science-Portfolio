"""US-EPA Air Quality Index (AQI) from PM2.5 / PM10 concentrations (pure Python).

This module is the interpretation-critical core for turning a pollutant
concentration into an AQI sub-index and a category. It has no third-party
dependency beyond the standard library, so it is always importable and is the
basis of the known-answer unit tests.

The AQI is computed with the standard EPA piecewise-linear formula. Within the
breakpoint band that contains the (truncated) concentration ``C_p``::

    AQI = (I_hi - I_lo) / (C_hi - C_lo) * (C_p - C_lo) + I_lo

where ``[C_lo, C_hi]`` is the concentration band and ``[I_lo, I_hi]`` the
corresponding AQI band. EPA truncates the concentration before the lookup
(PM2.5 to 1 decimal place, PM10 to an integer); we follow that so the published
boundary values reproduce exactly.

PM2.5 breakpoints (24-hour, micrograms/m3), EPA AQI technical assistance table::

    C_lo    C_hi    I_lo  I_hi   category
    0.0     12.0      0    50    Good
    12.1    35.4     51   100    Moderate
    35.5    55.4    101   150    Unhealthy for Sensitive Groups
    55.5   150.4    151   200    Unhealthy
   150.5   250.4    201   300    Very Unhealthy
   250.5   500.4    301   500    Hazardous

PM10 breakpoints (24-hour, micrograms/m3)::

    C_lo    C_hi    I_lo  I_hi   category
    0       54        0    50    Good
    55     154       51   100    Moderate
   155     254      101   150    Unhealthy for Sensitive Groups
   255     354      151   200    Unhealthy
   355     424      201   300    Very Unhealthy
   425     604      301   500    Hazardous

AQI category ranges (apply to the overall AQI = max of the sub-indices)::

    0-50      Good
    51-100    Moderate
    101-150   Unhealthy for Sensitive Groups
    151-200   Unhealthy
    201-300   Very Unhealthy
    301+      Hazardous

Worked, hand-derivable values (checked by the tests):

* ``pm25_to_aqi(12.0) == 50``  (top of the Good band).
* ``pm25_to_aqi(35.4) == 100`` (top of the Moderate band).
* ``pm25_to_aqi(9.0) == 38``   ((50/12)*9 = 37.5, rounded half-up to 38).
* ``pm10_to_aqi(54) == 50``    (top of the Good band).
* ``pm10_to_aqi(155) == 101``  (bottom of the USG band).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

# Each breakpoint row: (C_lo, C_hi, I_lo, I_hi). Concentration in micrograms/m3.
_PM25_BREAKPOINTS: list[tuple[float, float, int, int]] = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

_PM10_BREAKPOINTS: list[tuple[float, float, int, int]] = [
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 604.0, 301, 500),
]

# Lower bound of each AQI category, in ascending order.
_CATEGORY_BANDS: list[tuple[int, str]] = [
    (0, "Good"),
    (51, "Moderate"),
    (101, "Unhealthy for Sensitive Groups"),
    (151, "Unhealthy"),
    (201, "Very Unhealthy"),
    (301, "Hazardous"),
]


def _round_half_up(value: float) -> int:
    """Round to the nearest integer, half away from zero (EPA convention)."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _truncate(conc: float, decimals: int) -> float:
    """Truncate (not round) a concentration to ``decimals`` places, as EPA does."""
    factor = 10**decimals
    # int() truncates toward zero; concentrations are non-negative here.
    return int(conc * factor) / factor


def _aqi_from_breakpoints(
    conc: float, breakpoints: list[tuple[float, float, int, int]], decimals: int
) -> int:
    """Apply the EPA piecewise-linear formula for one pollutant.

    The concentration is first truncated to ``decimals`` places (the EPA rule),
    then the band that contains it is found and the linear interpolation is
    applied and rounded half-up to an integer AQI.

    Raises
    ------
    ValueError
        If ``conc`` is negative or above the top of the table (off-scale).
    """
    if conc is None:
        raise ValueError("concentration must not be None.")
    conc = float(conc)
    if conc < 0:
        raise ValueError("concentration must be non-negative.")

    c_p = _truncate(conc, decimals)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= c_p <= c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (c_p - c_lo) + i_lo
            return _round_half_up(aqi)

    raise ValueError(
        f"concentration {conc} is above the top of the AQI table (off-scale)."
    )


def pm25_to_aqi(conc_ugm3: float) -> int:
    """AQI sub-index for a PM2.5 concentration in micrograms/m3.

    PM2.5 is truncated to one decimal place before the lookup. See the module
    docstring for the breakpoint table and worked values (e.g. ``12.0 -> 50``,
    ``35.4 -> 100``).
    """
    return _aqi_from_breakpoints(conc_ugm3, _PM25_BREAKPOINTS, decimals=1)


def pm10_to_aqi(conc_ugm3: float) -> int:
    """AQI sub-index for a PM10 concentration in micrograms/m3.

    PM10 is truncated to an integer before the lookup. See the module docstring
    for the breakpoint table and worked values (e.g. ``54 -> 50``, ``155 ->
    101``).
    """
    return _aqi_from_breakpoints(conc_ugm3, _PM10_BREAKPOINTS, decimals=0)


def aqi_from_pollutants(
    pm25: float | None = None, pm10: float | None = None
) -> int:
    """Overall AQI as the maximum of the available pollutant sub-indices.

    The EPA AQI for a location is the worst (largest) of its pollutant
    sub-indices. At least one of ``pm25`` / ``pm10`` must be given.

    Raises
    ------
    ValueError
        If both pollutants are ``None``.
    """
    sub_indices: list[int] = []
    if pm25 is not None:
        sub_indices.append(pm25_to_aqi(pm25))
    if pm10 is not None:
        sub_indices.append(pm10_to_aqi(pm10))
    if not sub_indices:
        raise ValueError("at least one of pm25 / pm10 must be provided.")
    return max(sub_indices)


def aqi_category(aqi: float) -> str:
    """Map an AQI value to its EPA category name.

    The bands are ``0-50`` Good, ``51-100`` Moderate, ``101-150`` Unhealthy for
    Sensitive Groups, ``151-200`` Unhealthy, ``201-300`` Very Unhealthy, and
    ``301+`` Hazardous. The lower edge of each band belongs to that band (e.g.
    AQI 101 is the first "Unhealthy for Sensitive Groups" value).

    Raises
    ------
    ValueError
        If ``aqi`` is negative.
    """
    if aqi is None:
        raise ValueError("aqi must not be None.")
    aqi = float(aqi)
    if aqi < 0:
        raise ValueError("aqi must be non-negative.")

    category = _CATEGORY_BANDS[0][1]
    for lower, name in _CATEGORY_BANDS:
        if aqi >= lower:
            category = name
        else:
            break
    return category
