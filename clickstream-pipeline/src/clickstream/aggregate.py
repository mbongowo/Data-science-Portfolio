"""Offline (batch) aggregations over a clickstream DataFrame (pandas only).

These functions reproduce the windowed counts a streaming job would emit, but
over a bounded pandas DataFrame, so results are reproducible and easy to test.
They depend only on numpy/pandas (never pyspark or kafka), so they run in the
same lightweight environment as the rest of the tested core. The streaming
equivalents live in :mod:`clickstream.pipeline`.
"""

from __future__ import annotations

import pandas as pd


def events_per_minute(df: pd.DataFrame) -> pd.Series:
    """Count events per one-minute bucket from an epoch-seconds ``ts`` column.

    Each row of ``df`` is one event. The ``ts`` column holds the event time in
    epoch seconds. Events are floored to the minute (``floor(ts / 60) * 60``,
    still in epoch seconds) and counted. The result is indexed by the minute
    bucket start and sorted ascending; minutes with no events are omitted (this
    is a count of observed events, not a reindexed dense series).

    Parameters
    ----------
    df:
        DataFrame with at least a ``ts`` column of epoch-second timestamps.

    Returns
    -------
    pandas.Series
        Integer counts indexed by minute-bucket start (epoch seconds), named
        ``count`` with index name ``minute``.

    Raises
    ------
    ValueError
        If ``df`` has no ``ts`` column.
    """
    if "ts" not in df.columns:
        raise ValueError("DataFrame must have a 'ts' column of epoch seconds.")

    ts = pd.to_numeric(df["ts"])
    minute = (ts // 60) * 60
    counts = minute.value_counts().sort_index()
    counts.index.name = "minute"
    counts.name = "count"
    return counts.astype("int64")
