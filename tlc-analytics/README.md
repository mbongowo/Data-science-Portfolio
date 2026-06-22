# tlc-analytics

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A tabular engine bake-off on NYC TLC trip records.** The same aggregation
workload — demand by hour and day of week, tipping by payment type, fare and
duration summaries — is run over multi-year, partitioned TLC Parquet (billions
of rows) through three engines, and they are ranked on runtime, memory, and
cost. No geospatial columns are used: lat/long are dropped on ingest and the
analysis is purely tabular.

---

## Result first

**Question.** Which engine runs the same aggregation workload over multi-year
TLC Parquet the **fastest and cheapest** — Spark, DuckDB, or a managed
warehouse?

**Answer (illustrative).** On a single machine over five years of yellow-taxi
data, **DuckDB wins for this workload**: in-process, vectorised, no cluster to
stand up. Spark only pulls ahead once the data outgrows one box and the cluster
is already warm. The warehouse is the least operational effort but bills per
query.

```
Engine      Runtime (s)   Peak mem (GB)   Cost / run     Notes
---------   -----------   -------------   -----------    --------------------------
duckdb           42            6.1        $0 (local)     single node, in-process
spark           110           14.0        $0 (local)*    JVM + shuffle overhead
warehouse        18            n/a        ~$0.04         scan-billed, no infra
---------   -----------   -------------   -----------    --------------------------
Recommendation: DuckDB for single-machine multi-year scans; Spark when the lake
outgrows one node; warehouse when you want zero infra and can accept per-query $.
* "local" Spark has no cluster cost but does not reflect a real distributed run.
```

Two insight lines from the marts themselves:

- **Tipping is a payment-type story.** Card trips carry essentially all the
  tips; cash tips are unrecorded, so mean `tip_pct` for cash is ~0 — an
  artefact of how the meter logs cash, not rider behaviour.
- **Demand is bimodal by hour.** Pickups peak in the evening commute and again
  late night on weekends; the hourly mart makes the two humps obvious.

*(All numbers above are illustrative placeholders; run the benchmark on your
hardware and dataset to regenerate them.)*

### What this bake-off does **not** let you conclude

- **Hardware-specific.** The ranking reflects one machine's CPU, RAM, and disk.
  A different box — more cores, faster NVMe — can reorder the engines.
- **Single-machine vs cluster.** "Local" Spark carries JVM and shuffle overhead
  with none of the distributed upside. A real multi-node cluster on a lake that
  does not fit on one machine is a different contest this does not measure.
- **Cold vs warm cache.** OS page cache and the warehouse's result cache swing
  timings several-fold. We discard a warm-up run and report the median, but the
  cache state is itself a result.
- **Dataset-size sensitivity.** The fastest engine at one month is not
  necessarily the fastest at five years. Crossover points exist; one run does
  not find them.
- **Workload-specific.** These are scan-and-group-by aggregations. Joins,
  window functions, or point lookups would rank the engines differently.

---

## How it works

```
data/README.md       # how to download TLC Parquet into a year=/month= lake
config/tlc.yaml      # parquet root, years, engines, queries, partition columns
        |
src/tlc/
  clean.py           # documented cleaning predicates + derived columns
  marts.py           # pandas groupby reference marts (the source of truth)
  benchmark.py       # time_callable, BenchmarkResult, summarize (ranking)
  engines.py         # run_duckdb / run_spark wrappers (lazy engine imports)
  cli.py             # `tlc` console entry point: `mart` and `benchmark`
```

The numeric core — the cleaning rules and the four aggregated marts — is pure
pandas/numpy with **no engine dependency**, so it is always importable and is
the basis of **hand-derived known-answer tests**: a tiny frame with planted bad
rows loses exactly those rows; a six-trip frame gives hand-counted hourly and
day-of-week demand, a mean card tip rate of 0.25, and a median fare of 25.0. The
pandas marts are the source of truth; the DuckDB-SQL and Spark runners in
`engines.py` express the identical logic and import `duckdb` / `pyspark` lazily,
so the core and the test suite run without the engine stack installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: the partitioned Parquet
layout, the same workload three ways, the benchmarking method, building the
mart, reading the insight charts, and the honest limitations.

---

## Run it

### Option A — pixi (recommended)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
# download TLC Parquet into data/raw/yellow/year=*/month=*/ (see data/README.md)
pixi run mart           # build the aggregated marts (pandas reference path)
pixi run benchmark      # run the engine bake-off
pixi run test
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make mart
make benchmark
make test
```

### Option C — Docker

```bash
docker build -t tlc-analytics .
docker run --rm tlc-analytics            # runs the test suite
```

The Docker image is a slim Python base that exercises the pure-pandas mart and
the test suite. Spark and warehouse runs happen outside this image.

---

## Configuration

Everything that changes *what* runs lives in [`config/tlc.yaml`](config/tlc.yaml):
the Parquet root and glob, the years in scope, the partition columns, the list
of engines under test, and the queries to benchmark.

---

## Data sources

- **NYC TLC trip records** — the Taxi & Limousine Commission publishes one
  Parquet file per service per month going back years; yellow taxi alone runs to
  billions of rows. Arranged as a Hive-partitioned (`year=/month=`) lake. **No
  geospatial columns are used** — pickup/dropoff coordinates and zone IDs are
  dropped on ingest. See [`data/README.md`](data/README.md) for download links
  and the directory layout.

Raw data and outputs are git-ignored and reproducible from the download steps.

---

## License

MIT © 2026 Joseph Mbuh
