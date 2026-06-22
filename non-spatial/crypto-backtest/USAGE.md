# Usage guide: the backtesting workflow

This guide walks one pass end to end: install the stack, ingest raw ticks to
partitioned Parquet, resample to OHLCV, compute indicators, specify a simple
strategy, run the no-look-ahead backtest with fees and slippage, read the
performance analytics, and check the data integrity and the look-ahead
guarantee. It closes with what the result does not establish.

The numeric core (`bars`, `indicators`, `integrity`, `engine`, `performance`)
runs with only numpy and pandas. The Polars/Spark resampling backends are for
large dumps and import lazily; they are not needed to follow most of this guide.

## 1. Install

The full stack (including Polars and PySpark, which also wants a JVM) resolves
most reliably through conda-forge. Pixi is the path the repository is set up for.

```bash
pixi install        # resolves dependencies and writes pixi.lock locally
pixi run test       # confirm the install: the test suite should pass
```

If you prefer pip:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A quick check that the core is importable without the heavy optional libraries:

```bash
python -c "import numpy, pandas; from backtest import backtest, resample_ohlcv; print('ok')"
```

## 2. Ingest ticks to partitioned Parquet

Download a Binance monthly trade archive (see `data/README.md` for the exact
URL pattern) and unzip it into `data/raw/`. The CSV columns are
`id, price, qty, quote_qty, time, is_buyer_maker, is_best_match`; only
`price`, `qty`, and `time` (epoch milliseconds) are used, and their names are
set in `config/strategy.yaml`.

Raw tick files are large, so the practical first move is to partition them to
Parquet (columnar, compressed, and far faster to re-read than CSV). For one
month the CLI does it in one step while resampling (next section). For many
months or symbols, resample each archive to Parquet bars and keep those; the
multi-gigabyte raw CSVs do not need to stay on disk once bars exist.

For dumps too large to hold in pandas, use the out-of-core backends in
`backtest.scale_pipeline`:

```python
from backtest.scale_pipeline import resample_ohlcv_polars

bars = resample_ohlcv_polars(
    "data/raw/BTCUSDT-trades-2024-*.csv", "1min",
    ts="time", price="price", size="qty", time_unit="ms",
)
bars.write_parquet("data/raw/BTCUSDT-1min.parquet")
```

`resample_ohlcv_spark` is the distributed equivalent for a cluster. Both produce
the same `ts, open, high, low, close, volume` schema as the pandas core.

## 3. Resample to OHLCV bars

The pandas core handles a single archive directly:

```bash
backtest bars --config config/strategy.yaml \
  --ticks data/raw/BTCUSDT-trades-2024-01.csv \
  --out data/raw/BTCUSDT-1min.parquet
```

Or in Python:

```python
import pandas as pd
from backtest.bars import resample_ohlcv

ticks = pd.read_csv("data/raw/BTCUSDT-trades-2024-01.csv")
ticks["time"] = pd.to_datetime(ticks["time"], unit="ms")
bars = resample_ohlcv(ticks, "1min", ts="time", price="price", size="qty")
```

Bars are **left-closed and left-labelled**: the bar stamped `09:00` aggregates
trades in `[09:00, 09:01)`. Intervals with no trades produce no bar (no silent
forward-fill); the integrity check finds the resulting gaps. Keeping each bar
labelled on its own interval — not the future — is what lets the engine act on
the next bar cleanly.

## 4. Compute indicators

All indicators are strictly trailing: the value at bar `t` uses only bars up to
`t`, and the leading bars are `NaN` until a full window exists. That is a
correctness requirement, not a cosmetic one — an indicator that peeks ahead
re-introduces look-ahead through the back door.

```python
from backtest.indicators import sma, ema, rsi, rolling_vol, macd, bollinger_bands, atr

close = bars["close"]
fast = sma(close, 10)
slow = sma(close, 30)
r14  = rsi(close, 14)                 # Wilder smoothing
vol  = rolling_vol(close.pct_change(), 30)

macd_df = macd(close, fast=12, slow=26, signal=9)   # macd / signal / histogram
bb      = bollinger_bands(close, n=20, k=2.0)        # middle / upper / lower
atr14   = atr(bars["high"], bars["low"], close, 14)  # Wilder ATR
```

RSI uses Wilder's moving average (an EMA with `alpha = 1/n`). A strictly
increasing series has no losses, so RSI is exactly 100 — the value the test
pins.

`macd` returns a three-column frame (`macd`, `signal`, `histogram`); on a
constant series all three are `0`. `bollinger_bands` returns `middle` (the SMA)
plus `upper`/`lower` at `k` population standard deviations; a flat series
collapses the bands onto the price. `atr` is the Wilder average of the true
range, where the true range uses the *previous* close (the first bar falls back
to `high - low`), so a gap is captured even when the bar's own range is small.
All three are strictly trailing — no peek ahead.

## 5. Specify a simple strategy

The default is an SMA crossover, long or flat:

```python
signal = (sma(close, 10) > sma(close, 30)).astype(float)   # 1 long, 0 flat
```

A signal is a *target position per bar*. Generating it from the bar's own close
is fine; what matters is that it is **executed on the next bar**, which the
engine enforces — you never have to remember to shift it yourself when calling
`backtest`.

## 6. Backtest with fees and slippage

```python
from backtest.engine import backtest, positions_from_signals

equity = backtest(close, signal, fee_bps=10.0, slippage_bps=5.0)
```

What the engine does, explicitly:

- **No look-ahead.** The position earning bar `t`'s return is `signal` from bar
  `t - 1` (`signals.shift(1)`, flat before the first trade). Inspect it with
  `positions_from_signals(signal)`. A signal on the bar where a spike happens
  cannot capture that spike; it can only act from the following bar.
- **Costs on every change.** Turnover is `|position.diff()|`; each change pays
  `(fee_bps + slippage_bps)` bps of traded size. Holding is free.
- **Equity curve.** Net per-bar returns are compounded into an equity curve that
  starts at `1.0`.

Worked micro-example (the engine test): prices `[100, 110, 121, 121]`, always
long, 15 bps cost. Positions are `[0, 1, 1, 1]`; one entry is charged once; the
equity path is `[1.0, 1.0985, 1.20835, 1.20835]`.

## 7. Performance analytics

```python
from backtest.performance import (
    total_return, sharpe, max_drawdown,
    sortino, calmar, win_rate, turnover, exposure,
)
from backtest.engine import positions_from_signals

rets = equity.pct_change().fillna(0.0)
positions = positions_from_signals(signal)

print(total_return(equity))                       # end-to-end growth
print(sharpe(rets, periods_per_year=525600))      # 1-min bars, 24/7 market
print(max_drawdown(equity))                        # worst peak-to-trough, positive
print(sortino(rets, 525600))                       # downside-only risk adjustment
print(calmar(equity, 525600))                      # CAGR / max drawdown
print(win_rate(rets))                              # fraction of up bars
print(turnover(positions))                         # avg per-bar position change
print(exposure(positions))                         # fraction of bars in the market
```

`sharpe` returns `0.0` when returns have no dispersion (a flat or constant
series) instead of dividing by zero. `max_drawdown` is the largest
`1 - equity / running_peak`, reported as a positive fraction
(`[100,120,90,150]` gives `0.25`). `sortino` penalises only *downside*
deviation and reports `+inf` when there is no downside and the mean is positive.
`calmar` is the annualised (CAGR) return over max drawdown. `win_rate`,
`turnover`, and `exposure` describe the *shape* of the strategy: how often it
wins, how much it trades (turnover is exactly what costs are charged on), and
how much of the time it is exposed. The whole report is also produced by the
CLI:

```bash
backtest run    --config config/strategy.yaml --bars data/raw/BTCUSDT-1min.parquet --out outputs
backtest report --config config/strategy.yaml --equity outputs/equity.parquet     --out outputs
```

which writes `outputs/equity.parquet` and a `summary.json` of the headline
numbers.

## 8. Data-integrity and look-ahead checks

Run the integrity checks **before** trusting any backtest:

```python
from backtest.integrity import find_gaps, find_duplicates, summarize_integrity

print(summarize_integrity(bars, "1min"))
# {'n_bars': ..., 'n_gaps': ..., 'n_duplicates': ..., 'coverage': ..., ...}

gaps = find_gaps(bars.index, "1min")        # missing minutes (outages, drops)
dups = find_duplicates(bars.index)          # repeated timestamps (overlap/double-write)
```

A series with silent gaps or duplicate bars yields a confident but wrong result.
Coverage near 1.0 with zero duplicates is the green light; anything else needs
explaining (an exchange maintenance window is fine, a truncated download is not).

The look-ahead guarantee is mechanical, but you can see it for yourself:
`positions_from_signals(signal)` is the signal shifted forward by one bar, so the
position over any bar depends only on prior information. The engine test asserts
that a signal generated on bar `t` leaves bar `t`'s equity untouched and only
bites from bar `t + 1`.

## 9. Validation discipline: walk-forward and cost sensitivity

A single in-sample number is the easiest thing to fool yourself with. Two cheap
guards live in `backtest.validation`.

**Walk-forward, out-of-sample windows.** `walk_forward_splits` yields
`(train, test)` index ranges that march forward in time. With the default
`step = test`, the test windows are non-overlapping and contiguous, and every
test index sits strictly after its own train window — so a strategy fit on
`train` is reported only on data it never saw.

```python
from backtest.validation import walk_forward_splits

for train_idx, test_idx in walk_forward_splits(len(close), train=300, test=100):
    fit  = close.iloc[list(train_idx)]   # tune/select on this
    oos  = close.iloc[list(test_idx)]    # report on this, untouched by the fit
    # ... fit on `fit`, evaluate on `oos` ...
```

**Cost sensitivity.** `sensitivity_sweep` re-runs the *real* engine across a
grid of fee and slippage assumptions and tabulates total return, so you can see
how fast the edge erodes as costs rise. The zero-cost row reproduces the gross
backtest exactly, and total return is (weakly) monotonically non-increasing as
either cost grows.

```python
from backtest.validation import sensitivity_sweep

sweep = sensitivity_sweep(
    close, signal,
    fee_grid=[0.0, 5.0, 10.0, 20.0],
    slippage_grid=[0.0, 5.0, 10.0],
)
print(sweep)   # columns: fee_bps | slippage_bps | total_return
```

If the result only survives at an optimistic fee, that is exactly what the
sweep is for. See [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb)
for both run end to end on the demo bars.

## 10. How to interpret responsibly

A passing backtest is evidence about one historical window, not a forecast.

**Past is not future.** Market regimes shift; a strategy that worked in a trend
can bleed in a range. Treat the result as a description of history.

**Overfitting is the default failure.** Searching windows, fees, and date ranges
until the equity curve looks good is curve-fitting dressed as research. Keep the
parameters in `config/strategy.yaml` so the choices are explicit, and prefer
out-of-sample windows you did not tune on.

**Survivorship bias.** Backtesting only on a symbol that is still trading ignores
the ones that were delisted or went to zero. The surviving series is the most
flattering and the least representative.

**Single asset, single venue.** One symbol on one exchange is one observation.
Nothing here shows the edge survives across assets, venues, or periods — and
fees, liquidity, and slippage all differ in ways a flat bps charge does not
capture.

**Costs are modelled, not exact.** A flat `fee + slippage` in basis points is a
deliberate simplification. Real fills face partial execution, market impact, and
fee tiers. If anything, the honest direction is to make costs harsher and see
whether the result survives.
