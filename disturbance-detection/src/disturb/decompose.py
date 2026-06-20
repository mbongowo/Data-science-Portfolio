"""Per-pixel seasonal-trend decomposition.

The core is a **pure-numpy harmonic regression**: we fit

    y(t) = a + b*t + sum_k [ c_k * cos(2*pi*k*f*t) + d_k * sin(2*pi*k*f*t) ]

by ordinary least squares, where ``f`` is the (annual) base frequency and
``k = 1..n_harmonics``. The fit decomposes a series into:

* ``trend``    = a + b*t                         (linear term)
* ``seasonal`` = sum of the harmonic (sin/cos) terms
* ``residual`` = y - trend - seasonal

Having a pure-numpy implementation means ``test_decompose`` runs with nothing
but numpy installed. An optional STL wrapper (statsmodels) is provided behind a
guarded import for users who want a non-parametric alternative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HarmonicFit", "harmonic_decompose", "stl_decompose"]


def _design_matrix(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Build the harmonic-regression design matrix.

    Columns: [1, t, cos(1), sin(1), cos(2), sin(2), ...].
    """
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t), t]
    base = 2.0 * np.pi / float(period)
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(base * k * t))
        cols.append(np.sin(base * k * t))
    return np.column_stack(cols)


@dataclass
class HarmonicFit:
    """Result of a harmonic decomposition for a single series.

    Attributes
    ----------
    t:
        Time coordinate used for the fit (in the same units as ``period``).
    trend, seasonal, residual:
        Additive components; ``observed`` of finite samples equals
        ``trend + seasonal + residual``.
    coeffs:
        Least-squares coefficients ``[a, b, c_1, d_1, c_2, d_2, ...]``.
    period:
        Base period of the seasonal cycle (e.g. 365.25 days).
    n_harmonics:
        Number of harmonic pairs fitted.
    """

    t: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    coeffs: np.ndarray
    period: float
    n_harmonics: int

    @property
    def intercept(self) -> float:
        return float(self.coeffs[0])

    @property
    def slope(self) -> float:
        """Linear trend slope (per unit of ``t``)."""
        return float(self.coeffs[1])

    def seasonal_amplitude(self, harmonic: int = 1) -> float:
        """Amplitude sqrt(c^2 + d^2) of the given harmonic (1-indexed)."""
        if harmonic < 1 or harmonic > self.n_harmonics:
            raise ValueError(f"harmonic must be in 1..{self.n_harmonics}")
        c = self.coeffs[2 * harmonic]
        d = self.coeffs[2 * harmonic + 1]
        return float(np.hypot(c, d))

    def seasonal_phase(self, harmonic: int = 1) -> float:
        """Phase (radians) of the given harmonic, as atan2(d, c)."""
        if harmonic < 1 or harmonic > self.n_harmonics:
            raise ValueError(f"harmonic must be in 1..{self.n_harmonics}")
        c = self.coeffs[2 * harmonic]
        d = self.coeffs[2 * harmonic + 1]
        return float(np.arctan2(d, c))

    def predict(self, t: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the fitted model (trend + seasonal) at ``t``.

        Parameters
        ----------
        t : numpy.ndarray, optional
            Time coordinate at which to evaluate the fit. Defaults to the ``t``
            used during fitting, in which case the result equals
            ``trend + seasonal``.

        Returns
        -------
        numpy.ndarray
            Fitted values ``a + b*t + sum_k harmonics`` at each ``t``.
        """
        if t is None:
            return self.trend + self.seasonal
        t = np.asarray(t, dtype=float).ravel()
        design = _design_matrix(t, self.period, self.n_harmonics)
        return design @ self.coeffs


def harmonic_decompose(
    series: np.ndarray,
    t: np.ndarray | None = None,
    period: float = 365.25,
    n_harmonics: int = 2,
) -> HarmonicFit:
    """Fit an additive trend + harmonic-seasonal model to ``series``.

    Parameters
    ----------
    series : numpy.ndarray
        1-D observations. NaNs are tolerated: they are excluded from the
        least-squares fit but components are still evaluated at every ``t``.
    t : numpy.ndarray, optional
        Time coordinate (same length as ``series``). If ``None``, an integer
        index ``0..n-1`` is used and ``period`` is interpreted in those units.
    period : float, optional
        Length of one seasonal cycle in the units of ``t`` (default 365.25,
        i.e. days per year).
    n_harmonics : int, optional
        Number of sin/cos harmonic pairs. The first harmonic captures the
        annual cycle; the second captures a semiannual asymmetry (green-up
        faster than senescence). 2-3 captures most vegetation cycles.

    Raises
    ------
    ValueError
        If ``series`` is empty, ``t`` length differs, ``n_harmonics < 1``, or
        too few finite samples remain to identify the model parameters.

    Returns
    -------
    HarmonicFit
        Trend, seasonal, residual components plus fitted coefficients.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if n == 0:
        raise ValueError("series is empty")
    if t is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.asarray(t, dtype=float).ravel()
    if t.size != n:
        raise ValueError("series and t must have the same length")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")

    design = _design_matrix(t, period, n_harmonics)

    finite = np.isfinite(y)
    if finite.sum() < design.shape[1]:
        raise ValueError(
            "not enough finite samples to fit the harmonic model "
            f"({int(finite.sum())} < {design.shape[1]} parameters)"
        )

    coeffs, *_ = np.linalg.lstsq(design[finite], y[finite], rcond=None)

    # Trend = intercept + slope * t (first two design columns).
    trend = design[:, :2] @ coeffs[:2]
    # Seasonal = remaining harmonic columns.
    seasonal = design[:, 2:] @ coeffs[2:]
    residual = y - trend - seasonal

    return HarmonicFit(
        t=t,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        coeffs=coeffs,
        period=float(period),
        n_harmonics=int(n_harmonics),
    )


def stl_decompose(series: np.ndarray, period: int = 23):
    """Optional STL decomposition via statsmodels (guarded import).

    STL is non-parametric and handles evolving seasonality, at the cost of
    requiring a regularly-sampled, gap-free series. Falls back with a clear
    error if statsmodels is not installed.

    Parameters
    ----------
    series:
        1-D regularly-sampled observations (no NaNs).
    period:
        Number of samples per seasonal cycle (e.g. ~23 for 16-day composites).

    Returns
    -------
    statsmodels.tsa.seasonal.DecomposeResult
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "stl_decompose requires statsmodels. Install it or use "
            "harmonic_decompose (pure numpy)."
        ) from exc

    y = np.asarray(series, dtype=float).ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("STL requires a gap-free series (no NaNs)")
    return STL(y, period=period, robust=True).fit()
