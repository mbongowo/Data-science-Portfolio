"""Data-integrity checks for a bar series.

A backtest is only as honest as its data. Three failures corrupt results
silently and are easy to miss:

* **Gaps** — intervals where bars are missing (an exchange outage, a download
  that dropped a chunk, or a symbol that simply did not trade). They break the
  assumption that consecutive rows are one bar apart.
* **Duplicates** — repeated timestamps (a double-written archive, an overlap
  when concatenating monthly dumps).
* **Outages** — long runs of missing bars, which are gaps large enough that
  carrying a position across them is unrealistic.

These functions are pure pandas and are pinned by known-answer tests.
"""

from __future__ import annotations

import pandas as pd


def find_gaps(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return the timestamps that are missing from ``index`` at cadence ``freq``.

    A complete series at ``freq`` would contain every timestamp from
    ``index.min()`` to ``index.max()`` on that grid. The gaps are the grid
    points that are absent from ``index``.

    Parameters
    ----------
    index:
        Observed (sorted or unsorted) bar timestamps.
    freq:
        The expected cadence as a pandas offset alias, e.g. ``"1min"``.

    Returns
    -------
    pandas.DatetimeIndex
        Missing timestamps, in order. Empty if the series is complete.
    """
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    full = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
    return full.difference(idx)


def find_duplicates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the timestamps that appear more than once in ``index``.

    Each duplicated timestamp is reported once, in sorted order.
    """
    idx = pd.DatetimeIndex(index)
    dup = idx[idx.duplicated(keep="first")]
    return pd.DatetimeIndex(dup).sort_values().unique()


def summarize_integrity(df: pd.DataFrame, freq: str) -> dict[str, object]:
    """Summarise the integrity of a bar frame indexed by timestamp.

    Parameters
    ----------
    df:
        Bars indexed by a ``DatetimeIndex``.
    freq:
        Expected cadence as a pandas offset alias.

    Returns
    -------
    dict
        ``n_bars``, ``n_gaps``, ``n_duplicates``, ``coverage`` (observed /
        expected bar count in ``[0, 1]``), ``start`` and ``end``. ``coverage``
        is 1.0 for a complete, gap-free, duplicate-free series.
    """
    idx = pd.DatetimeIndex(df.index)
    gaps = find_gaps(idx, freq)
    dups = find_duplicates(idx)

    if len(idx) == 0:
        expected = 0
        coverage = 0.0
    else:
        unique_sorted = idx.sort_values().unique()
        expected = len(
            pd.date_range(start=unique_sorted.min(), end=unique_sorted.max(), freq=freq)
        )
        coverage = (len(unique_sorted)) / expected if expected else 0.0

    return {
        "n_bars": int(len(idx)),
        "n_gaps": int(len(gaps)),
        "n_duplicates": int(len(dups)),
        "coverage": float(coverage),
        "start": idx.min() if len(idx) else None,
        "end": idx.max() if len(idx) else None,
    }
