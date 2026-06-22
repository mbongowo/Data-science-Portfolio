# log-anomaly

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Unsupervised log anomaly detection.** Collapse raw log lines to event
templates (Drain-lite masking), count templates per session to build an
event-count matrix, score each session with a transparent detector (PCA
reconstruction error, a robust Mahalanobis distance, template rarity, or a
z-score rule), and — where labels exist — measure precision / recall / F1 and
compare detectors by PR-AUC / ROC-AUC. A separate Spark + IsolationForest path
scales the same pipeline to the full corpus. What the numbers do and do not
prove is written out, not hand-waved.

---

## Result first

**Question.** Given only unlabelled HDFS logs, can we flag the anomalous blocks —
the ones an operator would want to look at — without training on labels?

**Answer.** Yes, with caveats. Templating reduces the log to a few event types;
the per-block event-count matrix is dominated by a low-rank "normal" subspace, so
a rank-``k`` PCA reconstruction error separates the rare event mixes. Flagging
the blocks above a quantile of the error recovers most labelled anomalies at a
usable precision.

The numbers below are **real and reproducible in under a second** — they come
from a small, seeded **synthetic** HDFS-like log (300 block sessions, ~15%
anomalous) built so the result is deterministic and runnable anywhere, including
CI, with only numpy / pandas / pyyaml + stdlib (no Spark, no scikit-learn). The
demo drives the *identical* templating + PCA detection + metrics core that the
full Spark pipeline runs on the real labelled Loghub HDFS_v1 set.

```
detector         : PCA reconstruction error, k=3, flag > 0.85 quantile
blocks scored    : 300        (seeded synthetic, labelled Normal/Anomaly)
templates found  : 8
true anomalies   : 45
precision        : 0.74
recall           : 0.69
F1               : 0.71
confusion matrix : tn=244  fp=11  fn=14  tp=31
```

**Reproduce:** `pixi run demo` (or `make demo`, or
`loganomaly demo`) — writes `outputs/templates.csv`, `outputs/scores.csv`,
`outputs/summary.json`. The separation is deliberately imperfect: a few benign
rare events become false positives and a subtle minority of anomalies (which only
*omit* normal events) are missed, which is why precision and recall sit below
1.0 — a defensible operating point, not a staged perfect score.

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
  features.py       # event_count_matrix; template_idf + session_rarity; count_invariants
  detect.py         # pca_reconstruction_error, mahalanobis_scores, zscore_anomalies, flag
  evaluate.py       # precision_recall_f1, confusion_matrix; pr_curve, roc_curve, auc
  spark_pipeline.py # Spark ingest/parse at scale + optional IsolationForest (lazy)
  cli.py            # `loganomaly parse|detect|evaluate` console entry point
```

The numeric core (templating, features, the detectors, the metrics) is a
pure-numpy / pandas / stdlib layer with no heavy dependency. It is covered by
**hand-derived known-answer tests**: the worked HDFS line
`Receiving block blk_123 src: /10.0.0.1:50010 size 4096` masks to
`Receiving block <*> src: <*> size <*>`; a rank-1 event-count block plus one
off-pattern row puts the outlier's PCA reconstruction error on top; the 1-D
Mahalanobis case `[0,0,0,0,10]` scores `[0.25, 0.25, 0.25, 0.25, 4.0]`; a
perfectly separated score set gives ROC-AUC = 1.0; the metric case
`tp=2, fp=1, fn=1` gives precision = recall = F1 = 2/3. The heavy engines (Spark
parsing of the full corpus, sklearn IsolationForest) live in `spark_pipeline.py`
behind lazy imports, so the core and the test suite run without Spark or
scikit-learn installed.

### Capabilities (pure-numpy core)

- **Templating** — `mask_line`, `template_id`: collapse raw lines to event
  templates (Drain-lite masking).
- **Features** — `event_count_matrix` (session x template counts);
  `template_idf` + `session_rarity` (inverse-frequency rarity weighting, a
  cheap label-free score); `count_invariants` (flag sessions whose per-template
  counts fall outside a learned per-column quantile band).
- **Detectors** — `pca_reconstruction_error` (distance from the low-rank
  "normal" subspace); `mahalanobis_scores` + `mahalanobis_threshold` (robust
  whitened distance via a covariance pseudo-inverse, stable under constant /
  collinear columns); `zscore_anomalies`; `flag` (quantile thresholding).
- **Evaluation** — `precision_recall_f1`, `confusion_matrix`; `pr_curve`,
  `roc_curve`, and `auc` (trapezoid rule) for threshold-free comparison.

Because the detectors emit a continuous score, they can be **compared directly
via PR-AUC / ROC-AUC** without committing to one threshold — a higher AUC is the
better ranker. `notebooks/01_walkthrough.ipynb` runs the demo and tabulates
PCA vs Mahalanobis vs rarity this way (on the seeded corpus Mahalanobis leads on
both AUCs). The labels are used only to *score* the ranking, never to fit it.

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
