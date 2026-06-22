# log-anomaly

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Unsupervised log anomaly detection.** Collapse raw log lines to event
templates (Drain-lite masking), count templates per session to build an
event-count matrix, score each session with a transparent detector (PCA
reconstruction error or a z-score rule), and — where labels exist — measure
precision / recall / F1. A separate Spark + IsolationForest path scales the same
pipeline to the full corpus. What the numbers do and do not prove is written
out, not hand-waved.

---

## Result first

**Question.** Given only unlabelled HDFS logs, can we flag the anomalous blocks —
the ones an operator would want to look at — without training on labels?

**Answer (illustrative).** Yes, with caveats. Templating reduces ~11M HDFS lines
to a few dozen event types; the per-block event-count matrix is dominated by a
low-rank "normal" subspace, so a rank-``k`` PCA reconstruction error separates
the rare event mixes. Flagging blocks above the 99th percentile of error
recovers most labelled anomalies at a usable precision.

![Placeholder reconstruction-error / PR plot](outputs/.gitkeep)
<!-- Running `make detect` then `make evaluate` writes outputs/metrics.json;
     render the error histogram / PR curve from the notebook and drop the PNG here. -->

```
detector         : PCA reconstruction error, k=3, flag > 99th percentile
blocks scored    : 575,061   (labelled Normal/Anomaly via anomaly_label.csv)
precision        : 0.88
recall           : 0.81
F1               : 0.84
confusion matrix : tn=558,123  fp=2,210  fn=3,190  tp=13,538
```

*(Numbers above are illustrative placeholders; run the pipeline to regenerate
them for the configured detector and threshold.)*

### What this analysis does **not** let you conclude

- **Rule-based, not learned semantics.** Masking + PCA flags *statistically*
  rare event mixes. It does not understand the logs. A rare-but-benign event
  (a one-off admin action) looks identical to a rare-and-bad one. The flags are
  candidates for triage, not verdicts.
- **Threshold trade-off.** Precision and recall move with the quantile / z cut.
  There is no free lunch: tighten the threshold and recall falls, loosen it and
  precision falls. Report the operating point, not a single headline number.
- **Label leakage risk.** HDFS labels are block-level and were assigned with
  knowledge of known failures. Tuning the threshold *against* those labels and
  then reporting on the same blocks overstates performance. Hold out a split, or
  treat `evaluate` as a sanity check rather than a benchmark claim.
- **Drift.** Templates and the "normal" subspace are fit on one corpus. New
  software versions emit new lines, which mask to new templates and inflate the
  error for entirely healthy blocks. The model needs re-fitting as logs evolve.
- **Sessionisation assumption.** Everything is conditional on grouping lines
  into the right sessions (here, per HDFS block). A wrong session key scrambles
  the counts and the detector with them.

---

## How it works

```
data/README.md            # how to obtain Loghub HDFS_v1 (logs + anomaly_label.csv)
        |
src/loganomaly/
  templating.py     # Drain-lite: mask_line (blk_/hex/ip:port/number -> <*>), template_id
  features.py       # event_count_matrix: session x template counts
  detect.py         # pca_reconstruction_error, zscore_anomalies, flag (pure numpy)
  evaluate.py       # precision_recall_f1, confusion_matrix
  spark_pipeline.py # Spark ingest/parse at scale + optional IsolationForest (lazy)
  cli.py            # `loganomaly parse|detect|evaluate` console entry point
```

The numeric core (templating, features, the PCA / z-score detectors, the
metrics) is a pure-numpy / pandas / stdlib layer with no heavy dependency. It is
covered by **hand-derived known-answer tests**: the worked HDFS line
`Receiving block blk_123 src: /10.0.0.1:50010 size 4096` masks to
`Receiving block <*> src: <*> size <*>`; a rank-1 event-count block plus one
off-pattern row puts the outlier's PCA reconstruction error on top; the metric
case `tp=2, fp=1, fn=1` gives precision = recall = F1 = 2/3. The heavy engines
(Spark parsing of the full corpus, sklearn IsolationForest) live in
`spark_pipeline.py` behind lazy imports, so the core and the test suite run
without Spark or scikit-learn installed.

See [`USAGE.md`](USAGE.md) for the end-to-end workflow: ingest, templating,
feature vectors, detection, evaluation, and a section on what these flags do not
prove.

---

## Run it

### Option A — pixi (recommended; conda-forge ships the JVM pyspark needs)

```bash
pixi install            # resolves deps and GENERATES pixi.lock (not committed)
pixi run parse          # Spark: mask + template raw logs -> event counts
pixi run detect         # score sessions with the configured detector
pixi run evaluate       # precision / recall / F1 against the HDFS labels
pixi run test
```

> Note: `pixi.lock` is **machine-generated**. It is not committed here; running
> `pixi install` creates it on your platform.

### Option B — pip / venv

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
make parse
make detect
make evaluate
make test
```

### Option C — Docker

```bash
docker build -t log-anomaly .
docker run --rm log-anomaly        # runs the test suite (the pure-numpy core)
```

The Docker image runs the dependency-free core and its tests. Full Spark parsing
of the multi-million-line corpus runs separately via the pixi/conda-forge env.

---

## Configuration

Everything analysis-defining lives in [`config/hdfs.yaml`](config/hdfs.yaml):
the input glob, the session-key regex, which masking passes are on, the detector
(`pca` or `zscore`) and its parameters (k, quantile / z), and the path to the
HDFS label file for evaluation.

---

## Data sources

- **Loghub** (https://github.com/logpai/loghub) — a free collection of public
  system log datasets for research: **HDFS**, **BGL**, **Thunderbird**,
  **OpenStack**, and more.
- **HDFS_v1** is the primary set here because it is the one that ships **per-block
  anomaly labels** (`anomaly_label.csv`), which is what makes precision / recall /
  F1 possible. The other sets run through the same `parse` / `detect` path but
  have no labels to evaluate against.

Raw logs and outputs are git-ignored; see [`data/README.md`](data/README.md) for
how to obtain HDFS_v1 with its labels.

---

## License

MIT © 2026 Joseph Mbuh
