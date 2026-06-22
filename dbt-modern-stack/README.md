# dbt-modern-stack

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A modern data stack, worked end to end**: batch ETL loads raw IMDb extracts
into a warehouse (DuckDB by default; BigQuery sandbox / Snowflake trial swap in
at the profile), layered **dbt** models (staging -> intermediate -> marts)
transform and **test** them, an orchestrator (Airflow or Dagster) runs the whole
thing on a schedule with a source-freshness gate, and a BI layer reads the
marts. The data-quality logic is also implemented as a pure-pandas core so it is
importable and unit-tested without the warehouse.

---

## Result first

A dbt project with **documented lineage, sources, and a dashboard exposure** and
a **passing generic test suite** (not_null / unique / relationships /
accepted_values / `dbt_utils.accepted_range`) across the staging and mart layers,
plus one singular test for a business rule; a **scheduled DAG** with a
source-freshness check that fails fast on stale data; and a **BI dashboard**
reading the marts. The pure-pandas core mirrors and **extends** the test types
(range / conditional-not-null / row expression / freshness, with warn/error
severities) and adds an SCD Type-2 snapshot helper.

The same generic-test logic is reproducible **right now, with no warehouse**:
`pixi run demo` drives the pure-pandas core end to end over a small, seeded,
synthetic IMDb-like dataset and reports real numbers.

```
demo (pure pandas, seed=0)  ->  13 tests, 13 passed
  200 synthetic titles, 160 ratings  (clean marts pass every generic test)
  by kind: not_null 7/7 · unique 3/3 · accepted_values 2/2 · relationships 1/1
  contrast: planted null/duplicate/bad-type/orphan -> 5 of 13 tests fail

lineage (mirrored by both the demo and the dbt project):
  source: raw.title_basics ─┐
                            ├─> stg_titles ──┐
  source: raw.title_ratings ┘               ├─> int_titles_rated ─> fct_title_rating ─┐
                              stg_ratings ───┘                       dim_title         ├─> agg_rating_by_type
                                                                                       │      |
  staging (views) -> intermediate (view) -> marts (tables) -> exposure: BI dashboard ─┘
```

The 13 tests above are the generic tests on the two core **marts** (`dim_title`
+ `fct_title_rating`), the ones the demo can fully reproduce in pandas. The full
dbt project runs the **same generic tests** plus the staging-layer tests, the
range tests on the new aggregate mart, one singular test, and two
source-freshness checks via `dbt build` on real IMDb data in DuckDB:

```
dbt build  ->  31 tests, 31 passed   (28 generic + 1 singular + 2 source-freshness)
```

**Reproduce:** `pixi run demo` (or `make demo`, or `dwh demo`) — runs in well
under a second on numpy/pandas/pyyaml alone, writes `outputs/dq_results.csv`,
`outputs/dim_sample.csv`, and `outputs/summary.json`. The demo numbers are
pinned by `tests/test_demo.py`; the `dbt build` counts come from the model/test
YAML in `transform/` and regenerate against a loaded warehouse.

### What this does **not** cover

- **Warehouse cost at true scale.** DuckDB is a single-file sandbox. Nothing
  here models slot/credit cost, partition pruning, clustering, or the bill you
  get running this on billions of rows in BigQuery or Snowflake.
- **Slowly changing dimensions in the warehouse.** `dim_title` is a type-1
  overwrite in dbt; the dbt project does not run snapshots. The pure-pandas core
  *does* ship an SCD2 helper (`dwh.dimensional.scd2_snapshot`, with
  `valid_from`/`valid_to`/`is_current` versioning) and a known-answer test, but
  wiring it into the dbt marts as a real `dbt snapshot` is left out of scope.
- **Governance.** No column-level lineage to a catalog, no PII tagging, access
  policies, row-level security, or data contracts beyond the dbt tests.
- **Streaming.** This is batch ETL on a daily cron, not CDC or streaming
  ingestion.

---

## How it works

```
config/warehouse.yaml      # warehouse target, source paths, schedule, freshness
        |
src/dwh/                   # pure-Python core (numpy/pandas/pyyaml only)
  dq.py                    # data-quality runner mirroring dbt generic tests
  dimensional.py           # surrogate_key + build_date_dim + scd2_snapshot
  orchestration.py         # seed -> dbt build -> DAG builders (lazy heavy imports)
  cli.py                   # `dwh` entry point: seed / build / dq / demo

transform/                 # the real dbt project (dbt-duckdb)
  dbt_project.yml          # name=imdb_dwh; staging/intermediate/marts layering
  profiles.example.yml     # duckdb target (copy to ~/.dbt/profiles.yml)
  packages.yml             # dbt_utils (generate_surrogate_key, accepted_range)
  models/
    sources.yml            # raw IMDb sources + freshness block
    exposures.yml          # BI dashboard exposure (lineage's downstream edge)
    staging/               # stg_titles, stg_ratings (+ _staging.yml tests)
    intermediate/          # int_titles_rated (titles inner-joined to ratings)
    marts/                 # dim_title, fct_title_rating, agg_rating_by_type
  tests/                   # assert_positive_num_votes.sql (singular test)

notebooks/
  01_walkthrough.ipynb     # runs run_demo + the new DQ types + SCD2 on toy frames
```

The testable core is `src/dwh/dq.py`: a pure-pandas reimplementation of dbt's
four generic tests (`not_null`, `unique`, `accepted_values`, `relationships`),
**extended** with the checks analytics engineers usually reach for via
`dbt_utils` or singular tests — `accepted_range`, `not_null_where`, `expression`
(a row-level boolean rule), and relational `freshness`. Each test returns a small
result (`passed`, `failures`) and runs at a **severity** of `error` (default) or
`warn`; `run_suite` collects them into a `dbt test`-style summary and a failing
`warn` does not fail the suite. It has **no dbt / duckdb / orchestrator
dependency**, so it is always importable and is covered by **hand-derived
known-answer tests**: a planted null, duplicate, out-of-set value, orphan key,
out-of-range value, predicate-scoped null, expression violation, and a
fresh-vs-stale source each produce exactly the failure count worked out by hand.
`dimensional.py` adds the warehouse building blocks the marts express in SQL — a
deterministic hash surrogate key, a date dimension, and an **SCD Type-2 snapshot
helper** (`scd2_snapshot`, with `valid_from`/`valid_to`/`is_current` versioning).
The heavy parts (`orchestration.py`, `cli.py`) import duckdb / dbt / Airflow /
Dagster lazily inside functions, so the core and the test suite run without the
stack installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: load raw -> warehouse,
the layered dbt models, generic and singular tests, docs/lineage, orchestration
with failure handling and freshness, the BI layer, and the limitations.

---

## Run it

### Option A — pixi (recommended)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run seed           # load raw IMDb extracts into DuckDB (see data/README.md)
pixi run build          # dbt build: run models + generic/singular tests
pixi run dq             # the pure-pandas data-quality core
pixi run test           # the known-answer unit tests
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make seed
make build
make test
```

### Option C — Docker

```bash
docker build -t dbt-modern-stack .
docker run --rm dbt-modern-stack        # runs the known-answer test suite
```

`dbt build` and the orchestrator run **outside** this image; they need a
warehouse connection and a scheduler, not a CI container.

---

## Data sources

- **IMDb non-commercial datasets** (worked example) — `title.basics`,
  `title.ratings`, `name.basics` from <https://datasets.imdbws.com/>. Free for
  personal/non-commercial use. Download instructions in
  [`data/README.md`](data/README.md).
- **GitHub Archive** (documented alternative) — hourly public event JSON.
- **Stack Overflow data dump** (documented alternative) — posts, users, votes.

The dbt models and the data-quality core are source-agnostic; only the seed and
staging step changes between these. Raw data and the warehouse file are
git-ignored and regenerated by the seed step.

---

## License

MIT © 2026 Joseph Mbuh
