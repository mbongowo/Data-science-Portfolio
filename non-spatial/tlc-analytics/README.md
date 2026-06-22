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

**Answer (projected).** On a single machine over five years of yellow-taxi
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

> **Honesty note.** The engine table above is **projected** — representative of
> the full run over billions of partitioned Parquet rows on real hardware. Spark,
> DuckDB, and a managed warehouse cannot be stood up inside a seconds-long,
> dependency-free demo, so those three rows are **not measured here**. Reproduce
> them yourself with `pixi run benchmark` on a real dataset + engine stack.
>
> **One row is real, though.** When `duckdb` is installed, `pixi run demo` runs a
> *measured* pandas-vs-DuckDB bake-off on the demo frame (the same hourly-demand
> aggregation, pandas groupby vs in-process DuckDB SQL) and writes the timed
> ranking to `outputs/engine_bakeoff.csv`. When DuckDB is absent the harness
> records a clean `engine unavailable` skip instead of breaking — so CI stays
> green either way.

### Measured demo insights (pandas reference path)

These numbers **are measured** — produced by `pixi run demo`, which runs the
*real* pure-pandas core (`clean_trips` then the four marts, timed by the real
benchmark harness) over a seeded synthetic NYC-taxi-like frame. They are
deterministic and pinned by a test, not placeholders:

```
Reproduce:  pixi run demo          (seed=0, 5005 rows in, 5000 after cleaning)

Peak demand hour     18:00   (evening commute — synthetic demand is hour-weighted)
Card mean tip rate   17.9%   of fare
Cash mean tip rate    0.0%   of fare  (meter does not log cash tips)
Mean fare           $26.49
```

Two insight lines the marts make obvious:

- **Tipping is a payment-type story.** Card trips carry essentially all the
  tips; cash tips are unrecorded, so mean `tip_pct` for cash is ~0 — an
  artefact of how the meter logs cash, not rider behaviour. The demo measures
  card 17.9% vs cash 0.0%.
- **Demand peaks at rush hour.** Pickups peak in the evening commute; the demo's
  hourly mart puts the peak at **18:00**.

*(The four engine-comparison rows are projected from the full run; the demo
insight numbers above are measured by `pixi run demo` and reproducible anywhere
with only numpy / pandas / pyyaml + stdlib — no engines, no download.)*

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
  partitions.py      # iter_partitions: pure year=/month= lake path logic (no I/O)
  benchmark.py       # time_callable, summarize, bake_off (real pandas-vs-DuckDB)
  engines.py         # run_duckdb / run_spark wrappers (lazy engine imports)
  demo.py            # seeded synthetic trips -> real core -> measured insights
  cli.py             # `tlc` console entry point: `demo`, `mart`, `benchmark`
```

**Capabilities (pure pandas/numpy reference core):**

- **Cleaning** — `clean_trips`: four documented predicates + `trip_minutes` /
  `tip_pct` derived columns.
- **Marts** — `hourly_demand`, `demand_by_dow`, `tip_rate_by_payment`,
  `fare_summary`, plus `trip_duration_buckets` (binned trip-minute
  distribution), `revenue_by_day` (summed fare + tip per calendar day), and
  `anomaly_flags` (IQR / Tukey-fence outlier flags on fare and duration).
- **Partition logic** — `iter_partitions(root, years, months)` yields the
  expected `year=/month=` lake paths as pure stdlib path arithmetic (no I/O), so
  it is trivially testable and is the one place that knows the naming convention.
- **Bake-off harness** — `time_callable`, `summarize`, and `bake_off`, which runs
  a **real, measured** pandas-vs-DuckDB comparison on the demo data when `duckdb`
  is importable and records an explicit `engine unavailable` skip otherwise.

The numeric core is pure pandas/numpy with **no engine dependency**, so it is
always importable and is the basis of **hand-derived known-answer tests**: a tiny
frame with planted bad rows loses exactly those rows; a six-trip frame gives
hand-counted hourly and day-of-week demand, a mean card tip rate of 0.25, and a
median fare of 25.0; one trip per duration band; revenue summed per day; and a
single planted fare outlier flagged by the Tukey fence. The pandas marts are the
source of truth; the DuckDB-SQL and Spark runners in `engines.py` express the
identical logic and import `duckdb` / `pyspark` lazily, so the core and the test
suite run without the engine stack installed.

A runnable tour is in [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb):
it imports the package, runs `run_demo(0)`, and shows the new marts plus the
bake-off table.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: the partitioned Parquet
layout, the same workload three ways, the benchmarking method, building the
mart, reading the insight charts, and the honest limitations.

---

## Run it

### Option A — pixi (recommended)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run demo           # reproducible pandas demo on seeded synthetic data (no download)
# download TLC Parquet into data/raw/yellow/year=*/month=*/ (see data/README.md)
pixi run mart           # build the aggregated marts (pandas reference path)
pixi run benchmark      # run the engine bake-off
pixi run test
```

> `pixi run demo` needs only numpy / pandas / pyyaml + stdlib — no engines, no
> Parquet download — and prints the measured insight numbers shown above. If
> `duckdb` happens to be installed it additionally runs a **real** pandas-vs-
> DuckDB bake-off on the demo frame and writes `outputs/engine_bakeoff.csv`;
> without DuckDB it records a clean `engine unavailable` skip and still passes.

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
