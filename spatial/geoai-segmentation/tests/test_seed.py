"""Tests for the seed utility (numpy-only path).

These run without torch: ``seed_everything`` seeds python's ``random`` and
numpy unconditionally and only touches torch if it imports.
"""

from __future__ import annotations

import os
import random

import numpy as np

from geoseg.seed import seed_everything


def test_seed_returns_seed():
    assert seed_everything(123) == 123


def test_numpy_reproducible_across_calls():
    seed_everything(123)
    a = np.random.rand(8)
    seed_everything(123)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)


def test_python_random_reproducible_across_calls():
    seed_everything(7)
    x = [random.random() for _ in range(5)]
    seed_everything(7)
    y = [random.random() for _ in range(5)]
    assert x == y


def test_different_seeds_diverge():
    seed_everything(1)
    a = np.random.rand(8)
    seed_everything(2)
    b = np.random.rand(8)
    assert not np.array_equal(a, b)


def test_sets_pythonhashseed_env():
    seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
