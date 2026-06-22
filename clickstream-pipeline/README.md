# clickstream-pipeline

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Event-time stream processing**: tumbling and sliding windows, sessionisation
by inactivity gap, conversion funnels, and watermarks for late data. The
windowing logic is a pure-Python reference that anyone can read and test; the
Kafka + Spark Structured Streaming layer runs the same logic at scale and is
kept behind a lazily imported boundary. The boundary conditions (window edges,
late events, session gaps) are written out, not hand-waved.

---

## Result first

**Question.** Over a stream of user actions, **how many users reach each step of
the funnel** `view -> search -> add_to_cart -> checkout -> purchase`, and how do
event volumes move minute by minute?

**Answer (illustrative).** The funnel is monotonically non-increasing by
construction: every user who checks out has already viewed. The drop between
adjacent steps is where attention belongs. Per-minute counts come from the same
event-time windows the streaming job emits, so the offline number and the
streamed number agree on a settled window.

<!-- Running the offline CLI writes counts to outputs/; render the funnel bar
     chart from the notebook and drop the PNG here. -->

```
events/min (settled)  : 0->3   60->1   120->2          (tumbling, window=60s)
funnel reach          : view=50  search=38  add_to_cart=21  checkout=12  purchase=7
session gap           : 1800s  -> 63 sessions across 50 users
watermark             : allowed lateness 120s; 4 events dropped as late
```

*(Numbers above are illustrative placeholders; run the pipeline on your feed to
regenerate them.)*

### What this does **not** let you conclude

- **Not causal.** A funnel drop says users did not advance, not why. Price, page
  load, or copy are plausible causes but none is tested here.
- **Conditional on the window and the watermark.** Change the window width, the
  slide, or the allowed lateness and the per-minute series and the late-event
  count move. A window is "settled" only after the watermark has passed it;
  earlier reads are partial.
- **Conditional on the session gap.** Sessionisation splits on inactivity. A
  30-minute gap and a 5-minute gap give different session counts from the same
  events; the gap is a modelling choice, not a fact.
- **Late and out-of-order data is real.** Events past the watermark are dropped
  (or would late-update an already emitted window). The dropped count is part of
  the result, not an error to hide.
- **Ordering is per key.** Events are keyed by user so one user's actions stay
  ordered; there is no global order across users, and cross-user "sequences" are
  not meaningful.

---

## How it works

```
data/README.md         # Wikimedia EventStreams firehose OR a synthetic generator
        |
src/clickstream/
  windows.py           # tumbling_counts, sliding_counts, sessionize, funnel  (pure Python)
  watermark.py         # is_late, advance_watermark                            (pure Python)
  aggregate.py         # events_per_minute over a pandas DataFrame             (pandas)
  cli.py               # `clickstream` console entry point (typer, lazy imports)
  pipeline.py          # Kafka produce/consume + Spark Structured Streaming (guarded)
```

The interpretation-critical core is `windows.py`, `watermark.py`, and
`aggregate.py`: plain Python plus numpy/pandas, no streaming engine. It is
covered by **hand-derived known-answer tests** whose expected values are written
into the test docstrings — a tumbling window of 10s over five events gives
`{0: 2, 10: 2, 20: 1}`; sessionising `[0,1,2,10,11]` with a 5s gap gives
`[0,0,0,1,1]`; the funnel example resolves to `[3, 2, 1]`; a watermark with 5s
allowed lateness advances `0 -> 95 -> 98` and refuses to move back on a late
event. These functions return point answers only.

The Kafka producer/consumer and the Spark Structured Streaming job live in
`pipeline.py`. Every `pyspark` and `kafka` import is **inside** a function, so
the module imports for free and neither the package's `__init__` nor the test
suite touches the streaming stack. The streaming job uses the same windows and
the same watermark idea as the pure core, just over an unbounded stream.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: install, generate or
fetch events, run the offline CLI, stand up Kafka + Spark, read the funnel and
the per-minute series, and a section on what these numbers do not prove.

---

## Run it

### Option A — pixi (recommended; conda-forge resolves pyspark + the JVM)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run test
pixi run produce        # publish events onto Kafka (needs the stack up)
pixi run dashboard      # Streamlit view of the funnel + per-minute counts
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
clickstream per-minute data/raw/events.csv
make test
```

The offline CLI and the test suite need only numpy, pandas, pyyaml, and typer.
The streaming extras (`pyspark`, `kafka-python`, ...) are only needed for the
Kafka/Spark path.

### Option C — Docker

```bash
docker build -t clickstream-pipeline .
docker run --rm clickstream-pipeline        # runs the test suite
```

The image runs the pure-Python core and its tests. The full streaming stack
(Kafka broker + Spark, plus a JVM) is meant to run via `docker-compose`, not
this single image.

---

## Configuration

Everything that defines how the stream is windowed lives in
[`config/clickstream.yaml`](config/clickstream.yaml): window and slide sizes (in
seconds), the session gap, the Kafka topic names, the ordered funnel steps, and
the watermark delay.

---

## Data sources

- **Wikimedia EventStreams** (primary) — a free, public Server-Sent-Events
  firehose of recent changes (`stream.wikimedia.org/v2/stream/recentchange`),
  no key required. Each change has a timestamp, a user, and a type, which map
  onto the `(ts, user, event)` schema and arrive genuinely out of order.
- **Synthetic generator** (offline, deterministic) — a small seeded script that
  draws per-user sessions and injects out-of-order timestamps, for reproducible
  tests and demos.

See [`data/README.md`](data/README.md) for the schema and the fetch/generate
recipes. Raw data and outputs are git-ignored and regenerated on demand.

---

## License

MIT © 2026 Joseph Mbuh
