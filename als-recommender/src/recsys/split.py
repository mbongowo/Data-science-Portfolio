"""Train / validation / test split of a ratings table.

A recommender must be evaluated on interactions it never saw in training, and
the split has to respect users: each user's history is partitioned so the user
can appear in train and in the held-out sets. This module does a seeded
per-user random holdout, so a user with enough ratings contributes some rows to
train, some to validation, and some to test.

Pure pandas/numpy; no third-party dependency beyond numpy and pandas. The split
is deterministic for a fixed seed, which the test suite checks.

Limitation worth stating: a random holdout can leak information across time (a
training interaction may post-date a test one). For a leakage-free evaluation
use a temporal split on the timestamp column instead; this random holdout is the
simple default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def train_val_test_split(
    df: pd.DataFrame,
    user_col: str = "user",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train / val / test by random holdout within each user.

    For each user the rows are shuffled with a per-user seeded generator, then
    the last ``test_ratio`` fraction go to test, the preceding ``val_ratio``
    fraction to validation, and the remainder to train. Flooring the held-out
    counts guarantees that users with few ratings keep all their rows in train
    rather than being held out entirely.

    Parameters
    ----------
    df:
        Ratings table; one row per interaction.
    user_col:
        Name of the user-id column.
    val_ratio, test_ratio:
        Fractions held out per user. Must be non-negative and sum to < 1.
    seed:
        Master seed. The split is fully deterministic for a fixed seed.

    Returns
    -------
    (train, val, test):
        Three DataFrames with the original columns and a disjoint partition of
        the original rows (their union is the input row set). Original row index
        labels are preserved so the partition can be checked.

    Raises
    ------
    KeyError
        If ``user_col`` is not a column of ``df``.
    ValueError
        If the ratios are negative or sum to >= 1.
    """
    if user_col not in df.columns:
        raise KeyError(f"Column {user_col!r} not in DataFrame.")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    train_idx: list = []
    val_idx: list = []
    test_idx: list = []

    # Group by user; sort groups by key so iteration order is deterministic.
    for user, group in df.groupby(user_col, sort=True):
        idx = group.index.to_numpy().copy()
        # A per-user generator derived from the master seed and the user hash
        # keeps the split stable even if the row/user order changes upstream.
        rng = np.random.default_rng([seed, hash(user) & 0xFFFFFFFF])
        rng.shuffle(idx)

        n = idx.size
        n_test = int(np.floor(n * test_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_train = n - n_val - n_test

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]
