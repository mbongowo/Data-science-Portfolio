"""Tests for the seeded end-to-end demo.

These pin the demo's metrics to committed values: ``run_demo(seed=0)`` is fully
deterministic (seeded numpy ``default_rng`` + pure numpy/pandas core), so the
README can quote these exact numbers. They also re-assert the engine's
no-look-ahead guarantee on the demo's own bars and signal.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backtest.demo import run_demo

# Committed reference metrics for seed 0 (numpy default_rng + pure core).
EXPECTED = {
    "num_ticks": 60000,
    "num_bars": 1000,
    "num_trades": 36,
    "total_return": -0.034999478633277126,
    "sharpe": -3.824707142451097,
    "max_drawdown": 0.1212982989965915,
    "buy_and_hold_return": 0.012639063697363007,
}


def test_run_demo_metrics_are_deterministic(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=tmp_path)

    assert result["num_ticks"] == EXPECTED["num_ticks"]
    assert result["num_bars"] == EXPECTED["num_bars"]
    assert result["num_trades"] == EXPECTED["num_trades"]
    for key in ("total_return", "sharpe", "max_drawdown", "buy_and_hold_return"):
        assert result[key] == pytest.approx(EXPECTED[key], rel=1e-9, abs=1e-12)


def test_run_demo_writes_artifacts(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)
    for name in ("bars.csv", "equity_curve.csv", "summary.json"):
        assert (tmp_path / name).exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 0
    # Synthetic 1s ticks resample to a clean 1-minute grid: no gaps/dupes.
    assert summary["integrity"]["n_gaps"] == 0
    assert summary["integrity"]["n_duplicates"] == 0
    assert summary["integrity"]["coverage"] == pytest.approx(1.0)


def test_demo_signal_cannot_affect_its_own_bar(tmp_path) -> None:
    """Reconstruct the demo's bars/signal and prove no look-ahead holds.

    The engine executes a signal from bar ``t`` on bar ``t + 1`` (positions are
    ``signals.shift(1)``). We verify the executed position over each bar is the
    previous bar's signal, so a signal can never move the equity of its own bar.
    """
    from backtest.bars import resample_ohlcv
    from backtest.demo import _BAR_RULE, _FAST, _SLOW, _synthesize_ticks
    from backtest.engine import backtest, positions_from_signals
    from backtest.indicators import sma

    ticks = _synthesize_ticks(0)
    bars = resample_ohlcv(ticks, _BAR_RULE, ts="ts", price="price", size="size")
    close = bars["close"].astype(float)
    signal = (sma(close, _FAST) > sma(close, _SLOW)).astype(float)

    position = positions_from_signals(signal)
    # Executed position over bar t == signal from bar t-1 (first bar flat).
    assert position.iloc[0] == 0.0
    assert np.array_equal(position.to_numpy()[1:], signal.to_numpy()[:-1])

    # Flip the signal on a single bar; equity up to and including that bar is
    # unchanged, because that bar's position came from the prior signal.
    t = 200
    flipped = signal.copy()
    flipped.iloc[t] = 1.0 - flipped.iloc[t]
    eq0 = backtest(close, signal, fee_bps=10.0, slippage_bps=5.0)
    eq1 = backtest(close, flipped, fee_bps=10.0, slippage_bps=5.0)
    assert np.allclose(eq0.to_numpy()[: t + 1], eq1.to_numpy()[: t + 1])
