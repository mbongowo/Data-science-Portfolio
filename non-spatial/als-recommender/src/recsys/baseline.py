"""Popularity baseline for recommendation.

The most honest yardstick for a recommender is not "is it good in absolute
terms" but "does it beat recommending the globally most popular items to
everyone". Popularity is non-personalised, trivial to compute, and often
surprisingly hard to beat on offline ranking metrics — which is exactly why it
belongs in the evaluation.

This is a pure-pandas/numpy reference with no third-party dependency beyond
numpy and pandas.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def popularity_scores(
    train_df: pd.DataFrame, item_col: str = "item"
) -> list[tuple[object, int]]:
    """Rank items by popularity (interaction count) in the training data.

    Parameters
    ----------
    train_df:
        Training interactions. Each row is one user-item interaction.
    item_col:
        Name of the item-id column.

    Returns
    -------
    list of (item_id, count)
        Items ordered by descending interaction count. Ties are broken by item
        id (ascending) so the ordering is deterministic.

    Raises
    ------
    KeyError
        If ``item_col`` is not a column of ``train_df``.
    """
    if item_col not in train_df.columns:
        raise KeyError(f"Column {item_col!r} not in DataFrame.")
    counts = train_df[item_col].value_counts()
    # value_counts already sorts by count desc; resolve ties by item id asc to
    # make the ranking fully deterministic regardless of input row order.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(item, int(count)) for item, count in ranked]


def recommend_popular(
    train_df: pd.DataFrame,
    k: int,
    item_col: str = "item",
    exclude: Iterable[object] | None = None,
) -> list[object]:
    """Return the top-``k`` most popular items, optionally excluding some.

    Parameters
    ----------
    train_df:
        Training interactions.
    k:
        Number of items to return.
    item_col:
        Name of the item-id column.
    exclude:
        Items to drop before taking the top-``k`` (e.g. items the user has
        already seen in training). Defaults to none.

    Returns
    -------
    list
        Up to ``k`` item ids, most popular first. The same list is recommended
        to every user (this baseline is non-personalised), minus anything in
        ``exclude``.

    Raises
    ------
    ValueError
        If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    drop: set[object] = set(exclude) if exclude is not None else set()
    ranked: Sequence[tuple[object, int]] = popularity_scores(train_df, item_col)
    out: list[object] = []
    for item, _count in ranked:
        if item in drop:
            continue
        out.append(item)
        if len(out) == k:
            break
    return out
