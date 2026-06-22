# crypto-backtest

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Event-driven backtesting done carefully.** Ingest raw exchange ticks, resample
them to OHLCV bars, compute indicators, and run a strategy through an engine
that *cannot* see the future. The point of this project is not a big return
number — it is a return number you can defend: no look-ahead, realistic fees and
slippage, and data-integrity checks run before anything is trusted.

---

## Result first

**Question.** How does a simple SMA-crossover strategy perform, *end to end*,
once you resample ticks to bars, delay execution to the next bar, and charge
realistic fees and slippage on every trade?

**Answer (real, reproducible).** The numbers below are produced by
[`backtest demo`](src/backtest/demo.py) — the *actual* core (integrity checks ->
resample -> indicators -> no-look-ahead engine -> performance) run on a small
**seeded synthetic** tick series. They regenerate byte-for-byte in seconds with
no external data, so they are pinned in the test suite.

```
SMA(10/30) crossover, long/flat, on 1000 one-minute bars (60,000 synthetic ticks)
-----------------------------------------------------------------------------------
total return     : -3.50%
Sharpe (annual)  : -3.82     (periods_per_year = 525600, 1-min bars)
max drawdown     : 12.13%
trades           : 36        fee + slippage = 10 + 5 = 15 bps per position change
buy & hold       : +1.26%    (baseline, same bars, no costs)

Assumptions: signal on bar t EXECUTES on bar t+1 (no look-ahead); indicators
strictly trailing; data integrity checked first (0 gaps, 0 duplicates,
coverage 1.0); price path = seeded synthetic geometric Brownian motion.
```

**Reproduce:** `pixi run demo` (or `make demo`, or
`backtest demo --seed 0 --out outputs`). Writes `outputs/bars.csv`,
`outputs/equity_curve.csv`, and `outputs/summary.json`.

**Read this honestly.** The price path is a seeded synthetic random walk with
*zero drift*, so a costed crossover is **expected to lose** — and it does,
underperforming buy-and-hold. That is the point: this demo proves the machinery
is rigorous and reproducible, **not** that the strategy has a real-world edge.
The very same `resample -> indicators -> backtest -> performance` path runs on
real Binance tick dumps via `backtest run`. Rigor over returns.

### Assumptions and limitations

**How look-ahead bias is avoided.** A signal computed on bar `t` (from that
bar's close) is **executed on bar `t + 1`** — the engine holds
`signals.shift(1)`, never the same-bar outcome. Indicators are strictly
trailing. So no decision uses information it would not have had in real time.

**Costs that are actually applied.** Every position change pays
`fee_bps + slippage_bps` basis points of traded size (default 10 + 5 = 15 bps).
Holding an unchanged position is free. There is no rebate, no maker/taker
modelling beyond a flat bps charge, and no market-impact term.

**Data-integrity checks come first.** Before any backtest, the bars are screened
for missing intervals (gaps), duplicate timestamps, and long outages. A series
with silent gaps or duplicates produces a confident-looking but wrong result, so
coverage is reported alongside performance.

### What this does **not** establish

- **Past is not future.** A good backtest is a statement about a historical
  window, not a prediction. Regimes change.
- **Overfitting.** Tuning windows, fees, or the date range until the curve looks
  good is curve-fitting. The parameters live in `config/strategy.yaml` precisely
  so the choice is explicit and auditable, not so it can be optimised in secret.
- **Survivorship.** Testing on a symbol that is still actively traded ignores the
  ones that were delisted or collapsed. Single-symbol results are the most
  flattering and the least representative.
- **Single asset.** One symbol is one draw. Nothing here claims the edge
  generalises across assets, venues, or periods.

---

## How it works

```
data/                       # Binance tick dumps land here (git-ignored)
        |
src/backtest/
  bars.py          # ticks -> OHLCV via pandas resample (left-closed bars)
  indicators.py    # sma / ema / Wilder rsi / rolling_vol + macd / bollinger / atr
  integrity.py     # find_gaps / find_duplicates / summarize_integrity
  engine.py        # backtest(): signals.shift(1) => NO look-ahead; fees+slippage
  performance.py   # total_return / sharpe / max_drawdown + sortino / calmar /
                   #   win_rate / turnover / exposure
  validation.py    # walk_forward_splits (out-of-sample) + sensitivity_sweep (costs)
  scale_pipeline.py# Polars + Spark resampling for huge dumps (lazy imports)
  cli.py           # `backtest` console entry point (bars / run / report)
```

**Capabilities at a glance.**

- **Indicators (strictly trailing):** SMA, EMA, Wilder RSI, rolling volatility,
  **MACD** (line / signal / histogram), **Bollinger bands** (middle / upper /
  lower), **Wilder ATR**.
- **Performance:** total return, Sharpe, max drawdown, **Sortino** (downside
  deviation), **Calmar** (CAGR / max drawdown), **win rate**, **turnover**,
  **exposure**.
- **Validation discipline:** **walk-forward splits** yielding non-overlapping
  out-of-sample windows, and a fee/slippage **cost-sensitivity sweep** that
  re-runs the real engine across a grid and tabulates total return.
- **Walkthrough:** [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb)
  runs the demo and shows the new indicators, the cost sweep, and the
  walk-forward windows end to end.

The numeric core (`bars`, `indicators`, `integrity`, `engine`, `performance`,
`validation`) is pure numpy/pandas with no heavy optional dependency, so it is
always importable and is pinned by **hand-derived known-answer tests**: a tiny
tick set resamples to a checked OHLCV bar; the RSI of a strictly increasing
series is exactly 100; a constant series gives an all-zero MACD; Bollinger band
width on `[1..5]` follows the population sigma; a Wilder ATR with `n=1` equals
the per-bar true range; the Sortino of a no-downside series is `+inf`; Calmar of
`[100,120,90,150]` over three periods is `2.0`; a constant-long backtest over a
known price path reproduces an equity curve computed by hand, including one fee
charge; `max_drawdown([100,120,90,150])` is exactly `0.25`; `walk_forward_splits`
tiles non-overlapping out-of-sample windows; and the zero-cost row of a
sensitivity sweep reproduces the gross backtest. The Polars/Spark backends in
`scale_pipeline.py` handle the
multi-gigabyte resampling real dumps need and import lazily, so the core and the
test suite run on numpy/pandas alone.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves Polars/Spark cleanly)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run demo           # end-to-end on a seeded synthetic series (no data needed)
pixi run bars           # resample data/raw ticks -> 1-minute Parquet bars
pixi run run            # run the no-look-ahead backtest -> outputs/
pixi run test
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make bars
make run
make test
```

### Option C — Docker

```bash
docker build -t crypto-backtest .
docker run --rm crypto-backtest        # runs the test suite
```

---

## Configuration

Everything that defines a run lives in
[`config/strategy.yaml`](config/strategy.yaml): symbols, the bar rule (e.g.
`1min`), indicator windows, fee and slippage in basis points, the strategy
parameters, and the annualisation factor for Sharpe.

---

## Data sources

- **Binance public trade/tick dumps** (primary) — free historical per-trade data
  at <https://data.binance.vision/> (see
  <https://github.com/binance/binance-public-data>). These are huge (gigabytes
  per symbol-month), which is exactly why the resampling and the Polars/Spark
  backends exist. See [`data/README.md`](data/README.md) for how to obtain the
  trade CSVs.
- **Equity minute bars** (alternative) — any clean OHLCV minute-bar source drops
  straight into `backtest run`; only the ingest step differs. The engine and
  analytics do not care whether the bars came from crypto ticks or equities.

Raw data and outputs are git-ignored and regenerated by the ingest and backtest
steps.

---

## License

MIT © 2026 Joseph Mbuh
