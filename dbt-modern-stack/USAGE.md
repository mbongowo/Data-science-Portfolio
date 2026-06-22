# Usage guide: the modern data stack, end to end

This guide walks one full pass: load raw IMDb data into the warehouse, build the
layered dbt models, run the generic and singular tests, read the lineage and
docs, schedule the run with a freshness gate, and point a BI tool at the marts.
It closes with what this design does not handle.

The pure-pandas core (`dwh.dq`, `dwh.dimensional`) runs with only numpy, pandas,
and pyyaml installed and is meant for fast, dependency-light checks and for
unit-testing the data-quality logic. The warehouse, dbt, and the orchestrator
need the full stack from `requirements.txt` / `pixi.toml`.

## 1. Install

```bash
pixi install        # resolves dependencies and writes pixi.lock locally
pixi run test       # confirm the install: the known-answer tests should pass
```

Or with pip:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A quick check that the core imports without dbt/duckdb:

```bash
python -c "import pandas; from dwh import run_suite, surrogate_key; print('ok')"
```

## 2. Load raw -> warehouse

Download the three IMDb extracts into `data/raw/` (see `data/README.md` for the
exact `curl` lines), then seed them into DuckDB:

```bash
dwh seed --config config/warehouse.yaml
```

This reads each gzipped TSV (tab-separated, `\N` for NULL, unquoted) and writes
one table per source into the `raw` schema of `data/warehouse.duckdb`. The
config names the files, the DuckDB path, the schedule, and the freshness
thresholds; everything that says "where the data lives" is there, not in code.

In a real deployment this step is an Airbyte/Fivetran connector or a Python
extract job. The shape is the same: land raw, untouched, in its own schema, and
transform downstream so the raw layer stays a faithful copy of the source.

## 3. Layered dbt models

The dbt project lives in `transform/`. Copy the profile and build:

```bash
cp transform/profiles.example.yml ~/.dbt/profiles.yml   # or use --profiles-dir
cd transform
dbt deps        # installs dbt_utils (generate_surrogate_key)
dbt build       # runs models then tests, in dependency order
```

The models are layered by folder, each layer with its own materialization:

| Layer | Folder | Materialized | Job |
|---|---|---|---|
| Staging | `models/staging/` | view | One clean, typed, renamed row per source row. No joins, no business logic. `stg_titles`, `stg_ratings`. |
| Intermediate | `models/intermediate/` | view | Reusable joins and reshaping. `int_titles_rated` inner-joins titles to ratings — the fact grain. |
| Marts | `models/marts/` | table | The entities BI reads. `dim_title` (one row per title, surrogate + natural key), `fct_title_rating` (one row per rated title, FK to the dim, plus a derived `rating_band`), and `agg_rating_by_type` (one row per title type: rated-title count, mean rating, total votes, high-band share — the aggregate the dashboard reads directly). |

Staging isolates source quirks, so a column rename or a type fix happens once.
Marts are materialized as tables because BI hits them repeatedly and wants them
fast; staging and intermediate stay views to avoid storing intermediate copies.
The surrogate key in `dim_title`/`fct_title_rating` is a hash of the natural key
(`dbt_utils.generate_surrogate_key`), mirrored in Python by
`dwh.dimensional.surrogate_key`.

## 4. Generic and singular tests

dbt has two kinds of test, and this project uses both.

**Generic tests** are declared in the model YAML and reused across columns:

- `not_null` — the column has no NULLs.
- `unique` — no value repeats (NULLs ignored).
- `accepted_values` — every value is in an allowed set (here, `rating_band` in
  `{high, medium, low}`).
- `relationships` — every `fct_title_rating.title_key` matches a
  `dim_title.title_key` (referential integrity; no orphan facts).
- `dbt_utils.accepted_range` — numeric bounds: `average_rating` in `[1.0, 10.0]`,
  `num_votes >= 1`, `high_band_share` in `[0.0, 1.0]` on the aggregate mart.

See `transform/models/staging/_staging.yml` and `models/marts/_marts.yml`.

**Singular tests** are one-off SQL files that pass when they return zero rows.
`transform/tests/assert_positive_num_votes.sql` asserts no rated title has a
non-positive vote count — a business rule that does not fit a generic test
without a package macro.

Run just the tests with `dbt test`; `dbt build` runs them interleaved with the
models so a failed test stops dependents.

### The same logic in pure pandas

`dwh.dq` reimplements the four generic tests over DataFrames so the semantics are
importable and unit-tested without a warehouse:

```python
import pandas as pd
from dwh.dq import TestSpec, run_suite

titles = pd.DataFrame({"title_id": ["tt1", "tt2", "tt2"]})  # planted duplicate
ratings = pd.DataFrame({"title_id": ["tt1", "tt9"]})        # planted orphan

summary = run_suite([
    TestSpec("not_null", titles, "title_id", table="stg_titles"),
    TestSpec("unique", titles, "title_id", table="stg_titles"),
    TestSpec("relationships", ratings, "title_id", table="fct_title_rating",
             parent=titles, parent_col="title_id"),
])
print(summary)   # one row per test: test, table, column, passed, failures, detail
```

`dwh dq --config config/warehouse.yaml` runs this against the loaded warehouse
and exits non-zero if any test fails — handy as a fast pre-dbt smoke check. The
failure semantics deliberately match dbt, including the edge cases (`unique` and
`relationships` ignore NULLs).

### Extended test types and severities

Beyond dbt's four built-ins, `dwh.dq` ships the checks engineers usually pull in
via `dbt_utils` or singular tests, each with the same `(passed, failures)` shape:

- `test_accepted_range(df, col, lo, hi)` — numeric bounds (mirrors
  `dbt_utils.accepted_range`); `inclusive=False` makes the bounds themselves fail.
- `test_not_null_where(df, col, predicate)` — `not_null` restricted to the rows a
  predicate selects (`dbt_utils.not_null_where`).
- `test_expression(df, expr_fn)` — a row-level boolean rule that must hold for
  every row (`dbt_utils.expression_is_true`), for cross-column checks like
  `end_year >= start_year`.
- `test_freshness(df, ts_col, max_age, now)` — relational source freshness: stale
  when the newest timestamp lags `now` by more than `max_age` (mirrors
  `dbt source freshness`).

Each `TestSpec` carries a `severity` of `"error"` (default) or `"warn"`. A
failing `warn` test still shows on its summary row but does not fail the suite;
`suite_failed(summary)` only looks at `error` tests, exactly like dbt's
`severity: warn`.

```python
from datetime import datetime, timedelta
from dwh.dq import TestSpec, run_suite, suite_failed

df = pd.DataFrame({"rating": [5.0, 99.0], "_loaded_at": ["2026-01-01", "2026-01-01"]})
summary = run_suite([
    TestSpec("accepted_range", df, "rating", lo=1.0, hi=10.0, severity="warn"),
    TestSpec("freshness", df, ts_col="_loaded_at",
             max_age=timedelta(days=2), now=datetime(2026, 1, 2)),
])
print(summary)            # includes a `severity` column
print(suite_failed(summary))  # True only if an *error* test failed
```

### Slowly changing dimensions (SCD2) in pandas

`dwh.dimensional.scd2_snapshot(current, previous, key, cols, valid_from)` advances
a Type-2 snapshot by one run: unchanged keys carry forward, changed keys get their
prior version closed (`valid_to` set, `is_current=False`) and a new current version
appended, and new keys are inserted — the same history `dbt snapshot` maintains.
It is pure pandas and covered by a hand-derived known-answer test (see
`notebooks/01_walkthrough.ipynb` for a worked two-run example).

## 5. Docs and lineage

```bash
cd transform
dbt docs generate
dbt docs serve        # opens the catalog + the DAG
```

The docs site renders the model and column descriptions from the YAML, the test
coverage per column, and the lineage graph: sources -> staging -> intermediate
-> marts. `sources.yml` documents the raw IMDb tables and their freshness
expectations, so the lineage starts at the source, not at the first model.

## 6. Orchestration

`dwh.orchestration` wraps the pipeline (`seed_warehouse` -> `run_dbt_build`) and
builds a DAG for either scheduler; pick whichever your platform runs.

```python
from dwh.orchestration import build_airflow_dag   # or build_dagster_job
dag = build_airflow_dag("config/warehouse.yaml")   # drop into Airflow's dags/
```

Both DAGs are ordered: **freshness check -> seed -> dbt build**.

**Failure handling.** The first task, `assert_sources_fresh`, raises if any raw
extract is missing or older than its threshold in `config/warehouse.yaml`. Since
seed and build are downstream, a stale source fails the run *before* anything
loads, instead of silently publishing yesterday's data. The Airflow variant adds
a retry with a delay for transient failures. dbt's own `dbt source freshness`
check encodes the same thresholds in `sources.yml`, so the gate exists at both
the orchestration and the warehouse layer.

**Schedule.** The cron (`schedule.cron`, default `0 6 * * *`) runs a couple of
hours after IMDb's daily publish window. `name_basics` refreshes less often and
gets a longer freshness window than the title tables.

## 7. The BI layer

BI reads only the marts, never staging or raw. The two marts are a star: connect
the tool (Metabase, Superset, Looker, Power BI) to the DuckDB file (or to your
BigQuery/Snowflake target), join `fct_title_rating.title_key` to
`dim_title.title_key`, and build from there:

- average rating and vote count by `title_type` and `start_year` (decade),
- the `rating_band` mix (`high`/`medium`/`low`) over time,
- top genres by mean rating among titles above a vote threshold.

Because the marts are tested tables with documented columns, the dashboard rests
on a contract: if an upstream change breaks `not_null`/`unique`/`relationships`,
`dbt build` fails in orchestration and the dashboard is never refreshed from bad
data.

## 8. What this does not handle

These are real limits; carry them with any result.

**Warehouse cost at true scale.** DuckDB is a single-file sandbox with no bill.
Move this to BigQuery or Snowflake on billions of rows and cost becomes the main
design constraint — partitioning, clustering, incremental models, and slot/credit
budgeting — none of which the sandbox forces you to confront.

**Slowly changing dimensions in the warehouse.** `dim_title` overwrites in place
(type 1) in dbt: a retitled or recategorised title loses its prior value. The
pure-pandas `dwh.dimensional.scd2_snapshot` demonstrates the Type-2 pattern
(effective-dated `valid_from`/`valid_to`/`is_current` history), but it is not yet
wired into the dbt marts as a real `dbt snapshot`, which would change the grain
and the join logic.

**Governance.** The tests are the only contract here. There is no catalog
integration, column-level lineage to downstream tools, PII tagging, access
policies, row-level security, or formal data contracts between teams.

**Batch only.** This is a daily-cron batch pipeline. Anything needing
low-latency freshness (CDC, streaming ingestion, micro-batches) is out of scope,
and the freshness thresholds assume a once-a-day source.

**Test coverage is not correctness.** Green generic tests prove structural
properties (no nulls, unique keys, valid foreign keys, allowed values). They do
not prove the numbers are *right* — a wrong-but-well-formed rating passes every
test. Reconciliation against a trusted total is a separate exercise.
