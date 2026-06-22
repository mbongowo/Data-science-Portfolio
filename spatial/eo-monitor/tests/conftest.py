"""Pytest fixtures: tiny synthetic data so tests need no network or geo stack."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tiny_reflectance() -> dict[str, np.ndarray]:
    """A 2x2 synthetic reflectance scene with named logical bands.

    Values are chosen so the resulting indices are clean fractions.
    """
    return {
        "red": np.array([[0.1, 0.2], [0.2, 0.3]]),
        "green": np.array([[0.3, 0.4], [0.2, 0.1]]),
        "nir": np.array([[0.5, 0.8], [0.6, 0.7]]),
        "swir": np.array([[0.2, 0.4], [0.3, 0.5]]),
    }


@pytest.fixture
def tiny_baseline_stack() -> np.ndarray:
    """A (time=4, y=2, x=2) baseline stack of an index for anomaly tests."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.5, scale=0.1, size=(4, 2, 2))
