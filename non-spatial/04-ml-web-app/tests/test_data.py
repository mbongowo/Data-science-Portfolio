"""Tests for the data layer: label encoding, feature matrix, stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from croprec.data import (
    FEATURE_COLUMNS,
    encode_labels,
    feature_matrix,
    stratified_split,
)


def test_encode_labels_round_trip_and_order():
    labels = np.array(["maize", "rice", "maize", "beans"])
    y, classes = encode_labels(labels)
    # Classes are sorted alphabetically and deterministic.
    assert list(classes) == ["beans", "maize", "rice"]
    # Round-trip: classes[y] reconstructs the original labels.
    assert list(classes[y]) == list(labels)


def test_feature_matrix_shape_and_columns():
    df = pd.DataFrame(
        {
            "N": [1.0, 2.0],
            "P": [3.0, 4.0],
            "K": [5.0, 6.0],
            "temperature": [20.0, 21.0],
            "humidity": [60.0, 61.0],
            "ph": [6.0, 6.1],
            "rainfall": [100.0, 110.0],
            "label": ["maize", "rice"],
        }
    )
    X = feature_matrix(df)
    assert X.shape == (2, 7)
    # Column order matches FEATURE_COLUMNS.
    assert np.allclose(X[0], [1.0, 3.0, 5.0, 20.0, 60.0, 6.0, 100.0])
    assert list(FEATURE_COLUMNS) == ["N", "P", "K", "temperature", "humidity",
                                     "ph", "rainfall"]


def test_feature_matrix_missing_column_raises():
    df = pd.DataFrame({"N": [1.0]})
    with pytest.raises(ValueError):
        feature_matrix(df)


def test_stratified_split_disjoint_complete_and_balanced():
    # 30 of class 0, 30 of class 1.
    y = np.array([0] * 30 + [1] * 30)
    train, test = stratified_split(y, (0.7, 0.3), seed=0)

    # Disjoint and complete.
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted(train.tolist() + test.tolist()) == list(range(60))

    # Each class present in each split, roughly at the requested share.
    for split in (train, test):
        present = set(y[split].tolist())
        assert present == {0, 1}
    assert len(train) == 42  # 21 + 21
    assert len(test) == 18   # 9 + 9


def test_stratified_split_deterministic():
    y = np.array([0, 1] * 25)
    a = stratified_split(y, (0.6, 0.4), seed=3)
    b = stratified_split(y, (0.6, 0.4), seed=3)
    for pa, pb in zip(a, b, strict=True):
        assert np.array_equal(pa, pb)


def test_stratified_split_guards():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        stratified_split(y, (0.5, 0.6), seed=0)  # does not sum to 1
    with pytest.raises(ValueError):
        stratified_split(y, (1.5, -0.5), seed=0)  # non-positive fraction
