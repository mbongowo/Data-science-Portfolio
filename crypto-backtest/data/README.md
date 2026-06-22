# Data

Raw inputs and pipeline outputs are git-ignored (see `data/raw/.gitignore` and
the repository `.gitignore`). Nothing in `data/raw/` is committed; you download
it and the ingest step regenerates the Parquet bars.

## Binance public trade dumps (primary source)

Binance publishes free historical data dumps, including raw per-trade (tick)
data, at:

- https://data.binance.vision/ (browse) — also linked from
  https://github.com/binance/binance-public-data

The "trades" datasets are the tick stream this project is built around. They
are large (a single liquid symbol is gigabytes per month), which is the whole
reason the resampling step exists.

### How to obtain trade CSVs

1. Pick a symbol and granularity, e.g. spot `BTCUSDT`, monthly trades.
2. Download a monthly archive, for example:

   ```
   https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip
   ```

3. Unzip into `data/raw/`. The CSV columns are:

   ```
   id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
   ```

   This project only needs `price`, `qty`, and `time` (epoch milliseconds);
   the column names are configured in `config/strategy.yaml`.

4. Resample to OHLCV bars:

   ```bash
   backtest bars --config config/strategy.yaml \
     --ticks data/raw/BTCUSDT-trades-2024-01.csv \
     --out data/raw/BTCUSDT-1min.parquet
   ```

## Equity minute bars (alternative source)

If you would rather not handle multi-gigabyte tick dumps, any source of clean
OHLCV minute bars works as a drop-in for the resampled output: free equity
minute bars (for example via a broker API or a vendor sample) can be written to
the same Parquet schema (`open, high, low, close, volume` indexed by
timestamp) and fed straight into `backtest run`. The engine and analytics do
not care whether the bars came from crypto ticks or equity minutes.

## Integrity

Whatever the source, run the integrity checks before trusting a backtest:
missing-bar gaps, duplicate timestamps, and exchange outages all corrupt
results silently. See `summarize_integrity` in `src/backtest/integrity.py` and
the USAGE guide.
