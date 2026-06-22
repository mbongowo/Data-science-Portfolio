"""Tests for the pure-numpy data utilities. numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from lcnet.data import band_stats, patch_features, stratified_split


def test_stratified_split_disjoint_and_complete():
    y = np.repeat(np.arange(4), 25)  # 100 samples, 4 classes
    train, val, test = stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    # Disjoint.
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    # Covers every index exactly once.
    union = np.sort(np.concatenate([train, val, test]))
    assert np.array_equal(union, np.arange(100))


def test_stratified_split_every_class_in_every_split():
    y = np.repeat(np.arange(4), 25)
    train, val, test = stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    for split in (train, val, test):
        assert set(np.unique(y[split])) == {0, 1, 2, 3}


def test_stratified_split_deterministic():
    y = np.repeat(np.arange(3), 30)
    a = stratified_split(y, seed=42)
    b = stratified_split(y, seed=42)
    for x, z in zip(a, b, strict=True):
        assert np.array_equal(x, z)


def test_stratified_split_proportions_roughly_match():
    y = np.repeat(np.arange(2), 50)  # 100 samples
    train, val, test = stratified_split(y, (0.6, 0.2, 0.2), seed=0)
    assert train.size == 60
    assert val.size == 20
    assert test.size == 20


def test_stratified_split_bad_fractions_raise():
    y = np.repeat(np.arange(2), 10)
    with pytest.raises(ValueError):
        stratified_split(y, (0.5, 0.2, 0.2))  # does not sum to 1
    with pytest.raises(ValueError):
        stratified_split(y, (0.6, 0.4))  # not three entries


def test_band_stats_known_answer():
    X = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    mean, std = band_stats(X)
    assert np.allclose(mean, [2.0, 10.0])
    # std of [0,2,4] is sqrt(8/3); column 1 is constant -> 0
    assert abs(std[0] - np.sqrt(8.0 / 3.0)) < 1e-9
    assert std[1] == 0.0


def test_patch_features_shape_and_values():
    # 2 bands, 2x2 patch. Band 0 = all 1.0 (mean 1, std 0);
    # band 1 = [0,1,2,3] (mean 1.5, std = sqrt(1.25)).
    patch = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ]
    )
    feats = patch_features(patch)
    # interleaved [mean0, std0, mean1, std1]
    assert feats.shape == (4,)
    assert abs(feats[0] - 1.0) < 1e-9
    assert abs(feats[1] - 0.0) < 1e-9
    assert abs(feats[2] - 1.5) < 1e-9
    assert abs(feats[3] - np.sqrt(1.25)) < 1e-9


def test_patch_features_requires_3d():
    with pytest.raises(ValueError):
        patch_features(np.zeros((4, 4)))
