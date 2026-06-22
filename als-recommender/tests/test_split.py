"""Known-answer / invariant tests for the per-user train/val/test split.

The split must be (a) a disjoint partition of the input rows whose union is the
whole input, (b) per-user (each user's held-out rows come only from that user's
own history), and (c) deterministic for a fixed seed. These are checked on a
small constructed frame.
"""

from __future__ import annotations

import pandas as pd

from recsys.split import train_val_test_split


def _ratings() -> pd.DataFrame:
    rows = []
    # 5 users, 10 interactions each => 50 rows, enough for a 10/10 holdout.
    for user in range(5):
        for item in range(10):
            rows.append({"user": user, "item": item, "rating": float((item % 5) + 1)})
    return pd.DataFrame(rows)


def test_partition_is_disjoint_and_complete() -> None:
    """Train / val / test indices are disjoint and cover every input row."""
    df = _ratings()
    train, val, test = train_val_test_split(
        df, user_col="user", val_ratio=0.1, test_ratio=0.1, seed=42
    )
    idx_train = set(train.index)
    idx_val = set(val.index)
    idx_test = set(test.index)

    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)
    assert idx_train | idx_val | idx_test == set(df.index)


def test_holdout_sizes_per_user() -> None:
    """With 10 rows/user and 10% ratios, each user holds out 1 val and 1 test."""
    df = _ratings()
    train, val, test = train_val_test_split(
        df, user_col="user", val_ratio=0.1, test_ratio=0.1, seed=42
    )
    assert len(val) == 5  # one per user
    assert len(test) == 5
    assert len(train) == 40
    # The held-out rows for a user belong to that user only.
    for frame in (val, test):
        for user, group in frame.groupby("user"):
            assert (group["user"] == user).all()


def test_deterministic_with_fixed_seed() -> None:
    """Same seed => identical partition; different seed => different partition."""
    df = _ratings()
    a = train_val_test_split(df, user_col="user", test_ratio=0.2, seed=1)
    b = train_val_test_split(df, user_col="user", test_ratio=0.2, seed=1)
    c = train_val_test_split(df, user_col="user", test_ratio=0.2, seed=2)

    assert set(a[2].index) == set(b[2].index)
    # A different seed should move at least some rows (overwhelmingly likely).
    assert set(a[2].index) != set(c[2].index)


def test_rejects_bad_input() -> None:
    df = _ratings()
    import pytest

    with pytest.raises(KeyError):
        train_val_test_split(df, user_col="missing")
    with pytest.raises(ValueError):
        train_val_test_split(df, user_col="user", val_ratio=0.6, test_ratio=0.6)
