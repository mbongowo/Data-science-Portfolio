# Data

This pipeline does not ship a dataset. The streaming layer needs a live event
feed; the offline core needs a CSV of events. Both are produced from one of the
two sources below. Raw data is git-ignored (`data/raw/` keeps only its
`.gitignore`) and regenerated on demand.

## Event schema

Every event, whether synthetic or fetched, is reduced to three fields:

| field   | type           | meaning                                  |
|---------|----------------|------------------------------------------|
| `ts`    | int (epoch s)  | event time in seconds since the epoch    |
| `user`  | string         | a stable identifier for the actor        |
| `event` | string         | the event/action name (a funnel step)    |

The pipeline keys Kafka messages by `user` so one user's events stay ordered on
a single partition.

## Source A — Wikimedia EventStreams (free public SSE firehose)

Wikimedia publishes a real-time Server-Sent-Events feed of edits and related
activity, with no key or signup:

  https://stream.wikimedia.org/v2/stream/recentchange

Each line is a JSON change event with a `timestamp`, a `user`, and a change
`type`. Map those to the schema above (`ts = timestamp`, `user = user`,
`event = type`) to get a genuine, continuously arriving stream for the windowing
and watermark logic. Late and out-of-order arrival happens naturally, which is
the point of the watermark.

Fetch a bounded sample to `data/raw/` with any SSE client, for example:

```bash
curl -N https://stream.wikimedia.org/v2/stream/recentchange \
  | head -n 5000 > data/raw/recentchange.ndjson
```

then reshape the NDJSON into the three-column CSV the offline CLI expects.

## Source B — synthetic generator (offline, deterministic)

For reproducible tests and demos, generate events locally instead. A generator
draws per-user sessions: a user appears, emits a few funnel steps a few seconds
apart, goes idle past the session gap, and may return later. Out-of-order
timestamps are injected on purpose so the watermark has something to reject.

```python
import csv
import random

random.seed(0)
steps = ["view", "search", "add_to_cart", "checkout", "purchase"]
rows = []
t = 0
for user in range(50):
    t += random.randint(0, 30)
    depth = random.randint(1, len(steps))     # how far this user gets
    for k in range(depth):
        t += random.randint(1, 20)
        rows.append((t, f"u{user}", steps[k]))

with open("data/raw/events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ts", "user", "event"])
    w.writerows(rows)
```

The resulting `data/raw/events.csv` feeds the offline CLI directly:

```bash
clickstream per-minute data/raw/events.csv
clickstream funnel data/raw/events.csv "view,search,add_to_cart,checkout,purchase"
```

To exercise the streaming path, publish the same rows onto Kafka with
`clickstream.pipeline.produce_events` (see `USAGE.md`).
