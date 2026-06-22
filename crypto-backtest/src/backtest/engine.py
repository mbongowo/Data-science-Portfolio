"""The backtest engine: vectorised, event-driven, and free of look-ahead.

The single most important property of a credible backtest is that a decision
can only use information available *before* it is acted on. This engine
enforces that mechanically:

    A signal generated on bar ``t`` is executed at bar ``t + 1``.

Concretely, the position held over the return of bar ``t`` is
``signals.shift(1)`` — the signal you computed on the previous bar's close.
You never trade on a bar using that same bar's outcome. This shift is the line
that prevents look-ahead; everything else is bookkeeping.

Costs are charged on every **position change**: turning a position on, off, or
flipping it incurs ``(fee_bps + slippage_bps)`` basis points of the traded
size. Holding an unchanged position is free. Costs are subtracted from the bar
return in which the trade happens.

All pure numpy/pandas, so the whole thing is covered by a hand-derived
known-answer test.
"""

from __future__ import annotations

import pandas as pd


def backtest(
    prices: pd.Series,
    signals: pd.Series,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """Run a long/short backtest and return the equity curve.

    Parameters
    ----------
    prices:
        Bar close prices, indexed by timestamp. Bar returns are
        ``prices.pct_change()``.
    signals:
        Target position per bar, aligned to ``prices`` (e.g. ``1`` long, ``0``
        flat, ``-1`` short). The signal on bar ``t`` is **executed on bar
        ``t + 1``**: internally the position series is ``signals.shift(1)`` with
        the leading value filled to ``0`` (flat before the first trade).
    fee_bps:
        Trading fee in basis points of traded size, charged on every position
        change.
    slippage_bps:
        Assumed slippage in basis points of traded size, charged on every
        position change.

    Returns
    -------
    pandas.Series
        The equity curve (cumulative net growth), starting at ``1.0`` on the
        first bar and indexed like ``prices``.

    Raises
    ------
    ValueError
        If ``prices`` and ``signals`` are not the same length.
    """
    if len(prices) != len(signals):
        raise ValueError("prices and signals must have the same length.")

    prices = prices.astype(float)
    signals = signals.astype(float)

    # No look-ahead: act on the NEXT bar. The position earning bar t's return
    # is the signal decided on bar t-1.
    position = signals.shift(1).fillna(0.0)

    # Asset return of each bar; the first bar has no prior close.
    asset_ret = prices.pct_change().fillna(0.0)

    # Gross strategy return: position held over this bar times the bar return.
    gross = position * asset_ret

    # Cost on position changes. The first change is measured against a flat
    # (0.0) starting position.
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])  # change from flat at the open
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    cost = turnover * cost_rate

    net = gross - cost
    equity = (1.0 + net).cumprod()
    equity.iloc[0] = 1.0  # normalise the curve to start at 1.0
    return equity


def positions_from_signals(signals: pd.Series) -> pd.Series:
    """Return the executed positions, i.e. ``signals`` shifted forward by one.

    Exposed so callers (and tests) can see exactly which bar a signal acts on:
    the position over bar ``t`` is the signal from bar ``t - 1``; the first bar
    is flat.
    """
    return signals.astype(float).shift(1).fillna(0.0)
