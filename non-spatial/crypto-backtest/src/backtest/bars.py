"""Resample a raw tick (trade) stream into OHLCV bars.

A tick is a single executed trade: a timestamp, a price, and a traded size.
Resampling collapses all ticks that fall inside a fixed time interval into one
bar with open / high / low / close prices and summed volume.

This is pure pandas with no third-party dependency beyond numpy/pandas, so it
runs in the test suite. The Polars / Spark equivalents for very large
multi-symbol dumps live in :mod:`backtest.scale_pipeline`.

Convention: bars are labelled and closed on the **left** edge of each interval.
A bar timestamped ``09:00`` with rule ``"1min"`` therefore aggregates trades in
``[09:00, 09:01)``. Keeping the label on the bar's *own* interval (not the
future) is what lets the engine act on the next bar without leaking
information.
"""

from __future__ import annotations

import pandas as pd

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def resample_ohlcv(
    ticks: pd.DataFrame,
    rule: str,
    *,
    ts: str | None = None,
    price: str = "price",
    size: str = "size",
) -> pd.DataFrame:
    """Resample a tick ``DataFrame`` into OHLCV bars.

    Parameters
    ----------
    ticks:
        Trade-level data. Prices come from ``price`` and traded sizes from
        ``size``. Timestamps come either from the index (a
        ``DatetimeIndex``) or from a named column via ``ts``.
    rule:
        A pandas offset alias such as ``"1min"``, ``"5min"`` or ``"1h"``.
    ts:
        Name of the timestamp column. If ``None`` (default), the frame's index
        is used and must be a ``DatetimeIndex``.
    price:
        Name of the price column. Default ``"price"``.
    size:
        Name of the traded-size column. Default ``"size"``.

    Returns
    -------
    pandas.DataFrame
        Bars indexed by the left edge of each interval, with columns
        ``open, high, low, close, volume``. Intervals containing no trades are
        dropped (no synthetic forward-fill); use the integrity checks to find
        the resulting gaps.

    Raises
    ------
    ValueError
        If required columns are missing or no usable timestamp is available.
    """
    if price not in ticks.columns:
        raise ValueError(f"Price column {price!r} not found in ticks.")
    if size not in ticks.columns:
        raise ValueError(f"Size column {size!r} not found in ticks.")

    frame = ticks
    if ts is not None:
        if ts not in ticks.columns:
            raise ValueError(f"Timestamp column {ts!r} not found in ticks.")
        frame = ticks.set_index(pd.DatetimeIndex(ticks[ts]))
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            "ticks must have a DatetimeIndex or a timestamp column passed via ts."
        )

    grouped = frame.resample(rule, label="left", closed="left")
    bars = pd.DataFrame(
        {
            "open": grouped[price].first(),
            "high": grouped[price].max(),
            "low": grouped[price].min(),
            "close": grouped[price].last(),
            "volume": grouped[size].sum(min_count=1),
        }
    )
    # Drop empty intervals (no trades): close is NaN there.
    bars = bars.dropna(subset=["close"])
    return bars[_OHLCV_COLUMNS]
