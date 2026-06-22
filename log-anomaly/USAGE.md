# Usage guide: the log anomaly workflow

This guide walks through one pass of unsupervised log anomaly detection with this
repository: install the stack, get the data, mask raw lines into event
templates, build per-session event-count vectors, score sessions with a detector,
and (where labels exist) measure precision / recall / F1. It closes with what
these flags do not establish.

The pure-numpy core (`mask_line`, `template_id`, `event_count_matrix`,
`pca_reconstruction_error`, `zscore_anomalies`, `flag`, `precision_recall_f1`,
`confusion_matrix`) runs with only numpy / pandas installed and is meant for
small problems and for checking your understanding. The scale-out parsing of the
full multi-million-line corpus and the optional IsolationForest detector live in
`spark_pipeline.py` and need the Spark / scikit-learn stack described below.

## 1. Install

PySpark pulls a JVM; that resolves most reliably through conda-forge. Pixi is the
path the repository is set up for.

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

A quick check that the numeric core is importable without Spark or sklearn:

```bash
python -c "import numpy; from loganomaly import mask_line; print(mask_line('blk_1 size 4096'))"
```

## 2. Get the data

The primary dataset is **Loghub HDFS_v1**, the only Loghub set that ships
per-block anomaly labels. Download `HDFS_v1.zip` and unpack it into
`data/raw/HDFS_v1/` so you have `HDFS.log` and `anomaly_label.csv`. Full
instructions are in [`data/README.md`](data/README.md). The default paths in
`config/hdfs.yaml` match that layout.

The same `parse` / `detect` steps work on the unlabelled Loghub sets (BGL,
Thunderbird, OpenStack); only `evaluate` needs labels.

## 3. Templating: collapse lines to event types

Logs are mostly boilerplate with a few variable tokens. `mask_line` replaces the
variable tokens — block ids, hex ids, IPv4 / `IPv4:port`, bare numbers — with
`<*>`, so lines that differ only in their values collapse to one template.

```python
from loganomaly.templating import mask_line, template_id

mask_line("Receiving block blk_123 src: /10.0.0.1:50010 size 4096")
# 'Receiving block <*> src: <*> size <*>'

table: dict[str, int] = {}
template_id("Receiving block blk_1 size 4096", table)   # 0
template_id("Receiving block blk_2 size 8192", table)   # 0  (same template)
template_id("PacketResponder blk_3 terminating", table) # 1  (new template)
```

The masking passes run in a fixed order (block ids, then hex, then IP / IP:port,
then bare numbers) so that a specific token class is not chewed up by the generic
number pass. Toggle the passes in `config/hdfs.yaml` under `templating.mask`.

Templating is the load-bearing modelling choice here. Mask too aggressively and
distinct events merge into one template (you lose signal); mask too little and
every line with a different number becomes its own template (the matrix explodes
and nothing repeats). Inspect the template table before trusting anything
downstream: a healthy HDFS corpus collapses to a few dozen templates.

## 4. Feature vectors: the event-count matrix

Group the template-id sequence per session (per HDFS block) and count. Each row
is a session, each column a template, each cell a count.

```python
from loganomaly.features import event_count_matrix

sessions = {"blk_a": [0, 0, 1], "blk_b": [2, 1, 1]}
X = event_count_matrix(sessions, n_templates=3)
# array([[2., 1., 0.],
#        [0., 2., 1.]])
```

At scale, do steps 3 and 4 with Spark instead of in Python:

```python
from loganomaly.spark_pipeline import parse_logs_to_counts

parse_logs_to_counts(
    glob="data/raw/HDFS_v1/HDFS.log",
    session_regex=r"(blk_-?\d+)",
    out_parquet="outputs/event_counts.parquet",
)
```

or, from the command line, `loganomaly parse --config config/hdfs.yaml`. The
Spark masking is identical to `mask_line`; only the engine changes.

## 5. Detect: score the sessions

Two unsupervised detectors, both transparent and label-free.

**PCA reconstruction error.** Normal sessions share a small number of event
mixes, so the centred event-count matrix is approximately low rank. Project each
row onto the top-`k` principal directions and measure what is lost. A session
whose event mix is rare does not fit that subspace and has a large residual.

```python
from loganomaly.detect import pca_reconstruction_error, flag

errors = pca_reconstruction_error(X, k=3)
anomalies = flag(errors, quantile=0.99)   # top ~1% of blocks by error
```

When `k` reaches the rank of the centred matrix the projection is exact and every
error is ~0; that is the property the tests pin down. Choosing `k` is a bias /
variance call: too small and normal variety leaks into the residual (false
positives), too large and the subspace swallows the anomalies (false negatives).

**z-score rule.** A blunt baseline on a single derived score (e.g. the total
event count per session): flag scores more than `z` standard deviations above the
mean.

```python
from loganomaly.detect import zscore_anomalies

anomalies = zscore_anomalies(X.sum(axis=1), z=3.0)
```

It is one-sided (only unusually high scores flag) and a zero-variance input flags
nothing. Use it as a sanity baseline; PCA is the better detector on this matrix.

Run it from the command line: `loganomaly detect --config config/hdfs.yaml`. The
detector and its parameters come from the `detect` block of the config.

## 6. Evaluate: score the flags against labels

HDFS_v1 ships `anomaly_label.csv` mapping each block to Normal / Anomaly. Join it
to the flags and compute the metrics.

```python
from loganomaly.evaluate import confusion_matrix, precision_recall_f1

tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
precision, recall, f1 = precision_recall_f1(y_true, y_pred)
```

The positive class is "anomaly". `loganomaly evaluate --config config/hdfs.yaml`
joins flags to labels and writes `outputs/metrics.json` with the precision,
recall, F1, and confusion matrix.

Read the confusion matrix, not just the F1. False positives are blocks you will
waste triage time on; false negatives are the failures you missed. Which one
hurts more depends on the operational cost, and that sets the threshold — not the
other way round.

## 7. How to interpret responsibly

These detectors describe statistical rarity in the event mix. They do not explain
it, and a few limits should travel with any result.

**They are rule-based, not semantic.** Masking + PCA flags rare event
combinations. A rare-but-benign block (a one-off admin task) is indistinguishable
from a rare-and-bad one. Treat a flag as a candidate for a human to look at, not
as a diagnosis.

**The result depends on the threshold.** Precision and recall move with the
quantile or z cut; there is no single correct operating point. Report the cut you
used and show the trade-off, rather than quoting one F1 as if it were intrinsic.

**Labels can leak.** HDFS labels were assigned with knowledge of known failures.
Tuning the threshold against those same labels and then reporting on the same
blocks overstates performance. Hold out a split, or treat `evaluate` as a sanity
check rather than a benchmark.

**Templates and the subspace drift.** Both are fit on one corpus. A new software
release emits new lines that mask to new templates and inflate the reconstruction
error of perfectly healthy blocks. The model needs periodic re-fitting; a flag
spike after a deploy is often drift, not an incident.

**Sessionisation is an assumption.** Everything rests on grouping lines into the
right sessions. The wrong session key scrambles the counts and the scores with
them. Validate the session regex against a sample before scaling out.
