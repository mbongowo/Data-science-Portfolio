"""The harmonic decomposition must recover a known annual cycle."""

from __future__ import annotations

import numpy as np
import pytest

from disturb.decompose import harmonic_decompose


def _synthetic_pixel(seed: int = 0):
    """trend + annual sine + small noise, sampled every 16 days for 4 years."""
    rng = np.random.default_rng(seed)
    period = 365.25
    t = np.arange(0, 4 * 365, 16, dtype=float)  # ~91 samples
    true_amp = 0.3
    true_phase = 0.7  # radians
    trend = 0.5 + 0.00005 * t
    seasonal = true_amp * np.sin(2 * np.pi * t / period + true_phase)
    noise = rng.normal(0.0, 0.01, size=t.size)
    y = trend + seasonal + noise
    return t, y, period, true_amp, true_phase


def test_recovers_seasonal_amplitude_and_phase():
    t, y, period, true_amp, true_phase = _synthetic_pixel()
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)

    # Amplitude of the first harmonic should match the injected one closely.
    assert fit.seasonal_amplitude(1) == pytest.approx(true_amp, abs=0.02)

    # Recover phase. We injected a sine: y = A*sin(w t + phi)
    #   = A*sin(phi)*cos(w t) + A*cos(phi)*sin(w t)
    # so coeffs are c = A*sin(phi) (cos term), d = A*cos(phi) (sin term),
    # and atan2(d, c) = atan2(cos phi, sin phi) = pi/2 - phi.
    expected_fit_phase = np.pi / 2 - true_phase
    recovered = fit.seasonal_phase(1)
    diff = np.angle(np.exp(1j * (recovered - expected_fit_phase)))
    assert abs(diff) < 0.1


def test_components_sum_to_observed():
    t, y, period, *_ = _synthetic_pixel(seed=1)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    reconstructed = fit.trend + fit.seasonal + fit.residual
    np.testing.assert_allclose(reconstructed, y, atol=1e-9)


def test_residual_is_small_for_clean_signal():
    t, y, period, *_ = _synthetic_pixel(seed=2)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    # Residual should be near the injected noise level (std ~ 0.01).
    assert np.nanstd(fit.residual) < 0.03


def test_handles_nan_gaps():
    t, y, period, true_amp, _ = _synthetic_pixel(seed=3)
    y = y.copy()
    y[::7] = np.nan  # punch holes (clouds)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    assert fit.seasonal_amplitude(1) == pytest.approx(true_amp, abs=0.03)


def _two_harmonic_pixel(seed: int = 0):
    """Annual + semiannual signal with distinct amplitudes and phases."""
    rng = np.random.default_rng(seed)
    period = 365.25
    t = np.arange(0, 4 * 365, 8, dtype=float)  # ~183 samples
    a1, p1 = 0.30, 0.7
    a2, p2 = 0.12, -1.1
    seasonal = (
        a1 * np.sin(2 * np.pi * t / period + p1)
        + a2 * np.sin(2 * np.pi * 2 * t / period + p2)
    )
    y = 0.5 + 0.00003 * t + seasonal + rng.normal(0.0, 0.01, size=t.size)
    return t, y, period, (a1, p1), (a2, p2)


def _wrap(x: float) -> float:
    """Wrap an angle difference into (-pi, pi]."""
    return float(np.angle(np.exp(1j * x)))


def test_recovers_two_harmonic_amplitudes():
    t, y, period, (a1, _), (a2, _) = _two_harmonic_pixel(seed=10)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    assert fit.seasonal_amplitude(1) == pytest.approx(a1, abs=0.02)
    assert fit.seasonal_amplitude(2) == pytest.approx(a2, abs=0.02)


def test_recovers_two_harmonic_phases():
    t, y, period, (_, p1), (_, p2) = _two_harmonic_pixel(seed=11)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    # For y = A*sin(w t + phi), atan2(d, c) = pi/2 - phi.
    assert abs(_wrap(fit.seasonal_phase(1) - (np.pi / 2 - p1))) < 0.05
    assert abs(_wrap(fit.seasonal_phase(2) - (np.pi / 2 - p2))) < 0.05


def test_underfitting_one_harmonic_leaves_structure():
    """One harmonic cannot absorb a strong semiannual term; residual grows."""
    t, y, period, *_ = _two_harmonic_pixel(seed=12)
    fit1 = harmonic_decompose(y, t=t, period=period, n_harmonics=1)
    fit2 = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    assert np.nanstd(fit1.residual) > np.nanstd(fit2.residual)


def test_residual_is_white_for_correct_model():
    """Lag-1 autocorrelation of the residual is near zero when the model fits."""
    t, y, period, *_ = _two_harmonic_pixel(seed=13)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    r = fit.residual - fit.residual.mean()
    lag1 = float(np.sum(r[:-1] * r[1:]) / np.sum(r * r))
    assert abs(lag1) < 0.2


def test_predict_matches_components():
    t, y, period, *_ = _two_harmonic_pixel(seed=14)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    np.testing.assert_allclose(fit.predict(), fit.trend + fit.seasonal, atol=1e-9)
    # Evaluating on a subset reproduces those fitted values.
    np.testing.assert_allclose(
        fit.predict(t[:5]), (fit.trend + fit.seasonal)[:5], atol=1e-9
    )


def test_too_few_finite_samples_raises():
    t = np.arange(20, dtype=float)
    y = np.full(20, np.nan)
    y[:3] = 1.0  # fewer finite points than free parameters
    with pytest.raises(ValueError):
        harmonic_decompose(y, t=t, period=365.25, n_harmonics=2)


def test_invalid_harmonic_index_raises():
    t, y, period, *_ = _two_harmonic_pixel(seed=15)
    fit = harmonic_decompose(y, t=t, period=period, n_harmonics=2)
    with pytest.raises(ValueError):
        fit.seasonal_amplitude(3)
    with pytest.raises(ValueError):
        fit.seasonal_phase(0)
