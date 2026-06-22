# Usage guide: the clickstream workflow

This guide walks through one pass of the pipeline: install the stack, get an
event feed, run the offline aggregations on a file, then stand up the Kafka +
Spark streaming path that runs the same logic on an unbounded stream. It closes
with what these numbers do not establish.

The pure-Python core (`tumbling_counts`, `sliding_counts`, `sessionize`,
`funnel`, `is_late`, `advance_watermark`, `events_per_minute`) runs with only
numpy and pandas installed and is meant for small problems, for reproducing a
streamed result on a file, and for checking your understanding. The Kafka and
Spark engine that real ingestion needs lives in `clickstream.pipeline`, which
imports `pyspark` and `kafka` lazily, so the core and the test suite run without
the streaming stack installed.

## 1. Install

The streaming stack (pyspark and its JVM, plus duckdb and pyarrow) resolves most
reliably through conda-forge. Pixi is the path the repository is set up for.

```bash
pixi install        # resolves dependencies and writes pixi.lock locally
pixi run test       # confirm the install: the test suite should pass
```

If you prefer pip, expect to provide a JVM for pyspark yourself, then:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

A quick check that the numeric core is importable without the streaming stack:

```bash
python -c "from clickstream import tumbling_counts, sessionize; print('ok')"
```

## 2. Get an event feed

Every event is reduced to three fields: `ts` (epoch seconds), `user` (a stable
id), and `event` (the action name). There are two ways to get a feed; both are
described with copy-paste recipes in [`data/README.md`](data/README.md).

- **Wikimedia EventStreams** is a free public SSE firehose. Each recent-change
  event has a timestamp, a user, and a type, and they arrive genuinely out of
  order, which is what makes the watermark earn its keep.
- **The synthetic generator** draws per-user sessions deterministically and
  injects out-of-order timestamps. Use it for reproducible demos and tests.

Either way, land a CSV with the three columns at `data/raw/events.csv`. The
offline CLI reads that file; the streaming path publishes the same rows to Kafka.

## 3. Offline aggregations (the CLI)

The offline path reproduces, over a bounded file, the counts a streaming job
would emit. It needs only the lightweight core, so it is the fastest way to see
and check results.

```bash
# Events per one-minute bucket (epoch-second bucket start -> count)
clickstream per-minute data/raw/events.csv

# Tumbling-window counts at a chosen width
clickstream tumbling data/raw/events.csv --window-s 60

# Session ids for one user, in time order
clickstream sessions data/raw/events.csv u0 --gap-s 1800

# Funnel reach counts for an ordered, comma-separated step list
clickstream funnel data/raw/events.csv "view,search,add_to_cart,checkout,purchase"
```

How to read each one:

- **per-minute / tumbling.** Each event lands in exactly one window
  `[start, start + width)`, half-open, with `start = floor(ts / width) * width`.
  An event whose timestamp is exactly on a boundary belongs to the *later*
  window. Empty windows are omitted; the output is a count of observed events,
  not a dense reindexed series.
- **sliding.** With a window wider than the slide, an event is counted in every
  window that covers it, so the same event contributes to several rows. Use it
  for a smoothed, overlapping view of volume; use tumbling for partition-style
  non-overlapping counts.
- **sessions.** Walking a user's sorted timestamps, the session id increments
  whenever the gap to the previous event *exceeds* `gap_s`. A gap of exactly
  `gap_s` does not split. The gap is a modelling choice: report it.
- **funnel.** A user reaches step *k* if the first *k* step names appear in that
  user's events **in order** (not necessarily adjacent). The reach counts are
  monotonically non-increasing because reaching step *k* implies reaching *k-1*.
  Read the *drops* between adjacent steps, not the absolute heights.

## 4. Watermarks and late data

Streaming over event time forces a decision: how long to wait for stragglers
before a window is final. The watermark encodes that.

```python
from clickstream import advance_watermark, is_late

wm = 0.0
for ts in stream_of_event_times:
    wm = advance_watermark(wm, ts, allowed_lateness_s=120)
    if is_late(ts, wm):
        ...  # event at/under the watermark: drop or late-update
```

- `advance_watermark` tracks `max_event_ts - allowed_lateness_s` and is
  monotonically non-decreasing: an out-of-order event below the running maximum
  never pulls it back.
- `is_late` flags an event at or before the current watermark. An event exactly
  on the watermark counts as late, matching the "no more events <= watermark"
  contract.

A window is only **settled** once the watermark has passed its end. Reading a
window before that gives a partial count. The number of events dropped as late
is a real part of the result; surface it rather than hiding it. Widening the
allowed lateness recovers more stragglers at the cost of holding windows open
longer.

## 5. The streaming path (Kafka + Spark)

The same logic runs on an unbounded stream in `clickstream.pipeline`. Stand up a
broker (the bundled `docker-compose` is the intended way), then publish and
process.

```python
from clickstream.pipeline import produce_events, stream_events_per_minute

produce_events(
    [{"ts": 0, "user": "u0", "event": "view"},
     {"ts": 7, "user": "u0", "event": "search"}],
    topic="clickstream.events",
)

query = stream_events_per_minute(
    topic="clickstream.events",
    watermark_delay="2 minutes",     # mirrors allowed_lateness_s = 120
    window_duration="1 minute",      # mirrors tumbling_counts(window_s=60)
    output_path="outputs/per_minute", # omit to stream to the console
)
query.awaitTermination()
```

The job parses the JSON `ts` as an event-time column, sets a Spark watermark of
`watermark_delay`, and counts over tumbling windows of `window_duration`. Those
two settings are the streaming analogues of `advance_watermark` and
`tumbling_counts`; on a settled window the streamed count matches what the
offline CLI produces for the same events. Messages are keyed by `user`, so one
user's events keep their order on a single partition; there is no global order
across users.

Everything analysis-defining — topic names, window and slide sizes, the session
gap, the funnel steps, and the watermark delay — lives in
[`config/clickstream.yaml`](config/clickstream.yaml). Treat those values as part
of the result.

## 6. How to interpret responsibly

These counts describe the stream. They do not explain it, and a few limits
should travel with any result.

**They are not causal.** A funnel drop says users did not advance from one step
to the next, not why. Price, latency, layout, and copy are plausible drivers but
none is tested here. A drop is a prompt for investigation, not its conclusion.

**The result depends on the window and the watermark.** Change the window width
or the slide and the per-minute series moves; change the allowed lateness and
the dropped-event count and the "settled" timing move with it. Report the window
and the watermark alongside any number, and remember that a window read before
the watermark passes it is partial.

**The result depends on the session gap.** Sessionisation splits on inactivity,
so the session count is a function of `gap_s`. A 30-minute gap and a 5-minute gap
tell different stories from identical events. The gap is a choice, not a given.

**Late and out-of-order data is the normal case.** Real streams deliver events
behind their event time. The watermark draws the line past which they are
dropped or late-update an already emitted window. The number dropped is a
result; do not paper over it.

**Ordering is per key, not global.** Keying by user preserves one user's order
on a partition but says nothing about order across users. Do not read cross-user
"sequences" into the stream.

**Offline and streamed agree only on settled windows.** The pure-Python core and
the Spark job use the same window and watermark definitions, so their numbers
match once a window is final. While a window is still open, the streamed count is
incomplete by design and need not equal the offline count.
