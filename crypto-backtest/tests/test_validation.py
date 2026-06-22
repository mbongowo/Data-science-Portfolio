"""Known-answer tests for the validation discipline.

* walk_forward_splits(10, 4, 2) gives three non-overlapping out-of-sample test
  windows: (0..4, 4..6), (2..6, 6..8), (4..8, 8..10).
* The zero-cost row of a sensitivity sweep reproduces the gross backtest, and
  total return is non-increasing as fee/slippage rise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import backtest
from backtest.performance import total_return
from backtest.validation import sensitivity_sweep, walk_forward_splits


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# walk_forward_splits
# ---------------------------------------------------------------------------


def test_walk_forward_default_step_is_non_overlapping() -> None:
    folds = list(walk_forward_splits(10, train=4, test=2))
    # Default step == test -> contiguous, non-overlapping test windows.
    assert [(list(tr), list(te)) for tr, te in folds] == [
        ([0, 1, 2, 3], [4, 5]),
        ([2, 3, 4, 5], [6, 7]),
        ([4, 5, 6, 7], [8, 9]),
    ]
    # Test windows tile indices 4..10 with no overlap and no gap.
    test_indices = [i for _, te in folds for i in te]
    assert test_indices == list(range(4, 10))


def test_walk_forward_is_out_of_sample() -> None:
    # Every test index is strictly after every train index in the same fold.
    for train_idx, test_idx in walk_forward_splits(20, train=8, test=4):
        assert min(test_idx) > max(train_idx)


def test_walk_forward_single_window() -> None:
    # n exactly == train + test -> exactly one fold.
    folds = list(walk_forward_splits(6, train=4, test=2))
    assert len(folds) == 1
    assert list(folds[0][0]) == [0, 1, 2, 3]
    assert list(folds[0][1]) == [4, 5]


def test_walk_forward_too_short_yields_nothing() -> None:
    # Not enough data for even one train+test window.
    assert list(walk_forward_splits(5, train=4, test=2)) == []


def test_walk_forward_validates_args() -> None:
    with pytest.raises(ValueError):
        list(walk_forward_splits(0, 1, 1))
    with pytest.raises(ValueError):
        list(walk_forward_splits(10, 0, 2))
    with pytest.raises(ValueError):
        list(walk_forward_splits(10, 4, 2, step=0))


# ---------------------------------------------------------------------------
# sensitivity_sweep
# ---------------------------------------------------------------------------


def test_sensitivity_zero_cost_matches_gross() -> None:
    prices = _series([100.0, 110.0, 121.0, 121.0])
    signals = _series([1.0, 1.0, 1.0, 1.0])
    sweep = sensitivity_sweep(prices, signals, fee_grid=[0.0], slippage_grid=[0.0])

    gross = total_return(backtest(prices, signals).to_numpy())
    assert len(sweep) == 1
    assert sweep["total_return"].iloc[0] == pytest.approx(gross, abs=1e-12)


def test_sensitivity_grid_shape_and_monotonic() -> None:
    prices = _series([100.0, 110.0, 121.0, 100.0, 120.0])
    # A flip-flopping signal so costs actually bite.
    signals = _series([1.0, 0.0, 1.0, 0.0, 1.0])
    fee_grid = [0.0, 10.0, 20.0]
    slip_grid = [0.0, 5.0]
    sweep = sensitivity_sweep(prices, signals, fee_grid, slip_grid)

    # One row per (fee, slippage) combination, fee-major ordering.
    assert len(sweep) == len(fee_grid) * len(slip_grid)
    assert list(sweep.columns) == ["fee_bps", "slippage_bps", "total_return"]
    assert sweep["fee_bps"].tolist() == [0.0, 0.0, 10.0, 10.0, 20.0, 20.0]

    # Higher total cost -> weakly lower total return.
    by_cost = sweep.copy()
    by_cost["cost"] = by_cost["fee_bps"] + by_cost["slippage_bps"]
    by_cost = by_cost.sort_values("cost")
    rets = by_cost["total_return"].to_numpy()
    assert np.all(np.diff(rets) <= 1e-12)


def test_sensitivity_rejects_empty_grid() -> None:
    prices = _series([100.0, 110.0])
    signals = _series([1.0, 1.0])
    with pytest.raises(ValueError):
        sensitivity_sweep(prices, signals, fee_grid=[], slippage_grid=[5.0])
