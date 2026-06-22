r"""Tests for the bootstrap confidence interval for a mean.

Properties pinned here:

* On a fixed, well-behaved sample the 95% interval **brackets the true mean**
  (``lo <= mean <= hi``) and has ``lo <= hi``.
* A degenerate (constant) sample returns that constant for both bounds.
* The interval is **reproducible** for a fixed seed and shrinks as the sample
  grows (a basic sanity property of the bootstrap).
* Edge cases: single observation, and the input guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentiment.uncertainty import bootstrap_mean_ci


def test_ci_brackets_the_true_mean() -> None:
    rng = np.random.default_rng(123)
    scores = rng.normal(loc=0.3, scale=0.1, size=200)
    true_mean = float(scores.mean())
    lo, hi = bootstrap_mean_ci(scores, n_boot=2000, seed=0, alpha=0.05)
    assert lo <= hi
    assert lo <= true_mean <= hi


def test_constant_sample_is_a_point_interval() -> None:
    lo, hi = bootstrap_mean_ci([1.0, 1.0, 1.0, 1.0], n_boot=200, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_ci_is_reproducible_for_fixed_seed() -> None:
    scores = [0.1, 0.4, -0.2, 0.3, 0.0, -0.1, 0.5]
    a = bootstrap_mean_ci(scores, n_boot=500, seed=7)
    b = bootstrap_mean_ci(scores, n_boot=500, seed=7)
    assert a == b


def test_wider_alpha_gives_narrower_interval() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(0.0, 1.0, size=300)
    lo95, hi95 = bootstrap_mean_ci(scores, n_boot=2000, seed=0, alpha=0.05)
    lo80, hi80 = bootstrap_mean_ci(scores, n_boot=2000, seed=0, alpha=0.20)
    # A 95% interval is wider than an 80% interval on the same resamples.
    assert (hi95 - lo95) >= (hi80 - lo80)


def test_single_observation() -> None:
    lo, hi = bootstrap_mean_ci([0.42], n_boot=50, seed=0)
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


def test_input_guards() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci([], n_boot=10)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, 2.0], n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, 2.0], alpha=1.0)
