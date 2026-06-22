"""Validation discipline: walk-forward windows and cost-sensitivity sweeps.

The two cheapest ways to fool yourself with a backtest are (1) tuning on the
same data you report on, and (2) quoting a return that only survives at an
optimistic fee. This module addresses both, with pure numpy/pandas so each
piece is pinned by a hand-derived known-answer test.

* :func:`walk_forward_splits` yields ``(train, test)`` index ranges that march
  forward in time. The *test* windows are non-overlapping and strictly
  out-of-sample (every test index sits after its own train window), so a
  strategy fit on ``train`` is reported only on data it never saw.
* :func:`sensitivity_sweep` re-runs the **real** engine across a grid of fee and
  slippage assumptions and tabulates the resulting total return, so you can see
  at a glance how fast the edge erodes as costs rise.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from backtest.engine import backtest
from backtest.performance import total_return


def walk_forward_splits(
    n: int,
    train: int,
    test: int,
    step: int | None = None,
) -> Iterator[tuple[range, range]]:
    """Yield ``(train_idx, test_idx)`` ranges for walk-forward validation.

    Parameters
    ----------
    n:
        Total number of observations (e.g. number of bars).
    train:
        Length of each in-sample training window.
    test:
        Length of each out-of-sample test window.
    step:
        How far to advance the window origin between folds. Defaults to ``test``,
        which makes the **test windows non-overlapping and contiguous** — the
        usual anchored-free walk-forward layout. A smaller step overlaps the test
        windows; a larger one leaves gaps.

    Yields
    ------
    (train_idx, test_idx):
        Two ``range`` objects. ``train_idx`` is ``[origin, origin + train)`` and
        ``test_idx`` is ``[origin + train, origin + train + test)``. Every test
        index is strictly greater than every train index in the same fold, so the
        split is genuinely out-of-sample. Iteration stops once a full test window
        would run past ``n``.

    Notes
    -----
    With ``n=10, train=4, test=2`` the folds are
    ``(0..4, 4..6)``, ``(2..6, 6..8)``, ``(4..8, 8..10)`` for the default
    ``step=test=2`` — three non-overlapping out-of-sample windows covering
    indices 4..10. A single-window case (e.g. ``n=6, train=4, test=2``) yields
    exactly one fold.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if train < 1 or test < 1:
        raise ValueError("train and test must both be >= 1.")
    if step is None:
        step = test
    if step < 1:
        raise ValueError("step must be >= 1.")

    origin = 0
    while origin + train + test <= n:
        train_idx = range(origin, origin + train)
        test_idx = range(origin + train, origin + train + test)
        yield train_idx, test_idx
        origin += step


def sensitivity_sweep(
    prices: pd.Series,
    signals: pd.Series,
    fee_grid: list[float] | np.ndarray,
    slippage_grid: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Tabulate total return across a grid of fee / slippage assumptions.

    For every ``(fee_bps, slippage_bps)`` pair in the Cartesian product of the
    two grids, this re-runs the **real** :func:`backtest.engine.backtest` on the
    given prices and signals and records the resulting total return. The output
    is a tidy :class:`pandas.DataFrame` with one row per cost combination:

        ``fee_bps | slippage_bps | total_return``

    rows ordered fee-major then slippage. Because it reuses the genuine engine,
    the zero-cost row (``fee_bps=0, slippage_bps=0``) reproduces the gross
    backtest exactly, and total return is (weakly) monotonically non-increasing
    as either cost rises — the property the known-answer test checks.

    Raises
    ------
    ValueError
        If either grid is empty, or ``prices`` and ``signals`` differ in length
        (propagated from the engine).
    """
    fees = np.asarray(list(fee_grid), dtype=float)
    slips = np.asarray(list(slippage_grid), dtype=float)
    if fees.size == 0 or slips.size == 0:
        raise ValueError("fee_grid and slippage_grid must each be non-empty.")

    rows: list[dict[str, float]] = []
    for fee in fees:
        for slip in slips:
            equity = backtest(
                prices, signals, fee_bps=float(fee), slippage_bps=float(slip)
            )
            rows.append(
                {
                    "fee_bps": float(fee),
                    "slippage_bps": float(slip),
                    "total_return": total_return(equity.to_numpy()),
                }
            )
    return pd.DataFrame(rows, columns=["fee_bps", "slippage_bps", "total_return"])
