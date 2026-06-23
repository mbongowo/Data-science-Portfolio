# 02-mlops-pipeline

[![CI](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/mbongowo/Data-science-Portfolio/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A model taken from notebook to a monitored service — and kept honest after it
ships.** Predict whether *tomorrow is a rain day* (precipitation ≥ 1 mm) for a
Cameroon weather station from recent weather, then close the MLOps loop: track
the training run, serve predictions over HTTP, and **watch the live inputs for
drift** so the model's slow decay is caught instead of ignored. The runnable,
CI-tested core is a pure-numpy ML + **drift-detection** stack (PSI / KS); MLflow
tracking, the containerized FastAPI service, and Evidently dashboards are lazy
wrappers with a **free local** path and an **opt-in Azure ML** path.

Inspired by [DataTalksClub/mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp),
re-pointed at this portfolio's own weather domain and built around a
drift-monitoring core that most portfolios leave out.

---

## Result first

**Question.** Can recent weather predict whether it rains tomorrow at a Cameroon
station — and if the model were live, would we *notice* when the incoming data
drifts away from what it was trained on?

**Answer.** Yes to both. The block below is **real and reproducible in under two
seconds** — the output of `python -m mlpipe.cli demo`, which drives the actual
pure-numpy core (feature engineering → logistic regression → metrics → PSI/KS
drift) on seeded synthetic weather. A *reference* regime trains and evaluates the
model on a time-ordered holdout; a *current* regime is then synthesized to be
**warmer and wetter**, and the drift report catches it.

```
Rain-day model (time-ordered holdout, 179 test days)
  test accuracy     0.8939
  test F1           0.6667
  test ROC-AUC      0.9379

Drift report (reference vs warmer/wetter current regime)
  features drifted  10 of 12   (every temperature & precipitation feature)
  max PSI           8.79        (>> 0.2 "major shift" threshold)
  NOT drifted       doy_sin, doy_cos   (seasonal encoding is unchanged — correct)
```

**Reproduce:** `python -m mlpipe.cli demo` — runs the real core on the seeded
data and writes `outputs/metrics.json` and `outputs/drift_report.csv`. These
numbers are asserted as committed values in `tests/test_demo.py`. The model is
strong (ROC-AUC 0.94) because rain genuinely depends on recent temperature and
season; the drift detector flags every weather feature because the planted shift
moves them all, while leaving the two day-of-year features untouched — exactly
the signal you want when deciding whether to retrain.

### What this result does **not** let you conclude

- **The data is synthetic.** The headline proves the *machinery* is correct and
  reproducible, not that real Douala weather is this predictable. Point `mlpipe
  train` at a real station CSV for that.
- **High ROC-AUC at one threshold.** Everything is reported at a 0.5 decision
  threshold; F1 (0.67) shows the precision/recall trade-off at that operating
  point, which you would tune to the cost of a missed vs false rain alert.
- **PSI thresholds are heuristics.** 0.1 / 0.2 are industry conventions, not laws.
  A flagged feature is a prompt to investigate, not an automatic retrain.

---

## The ML lifecycle this closes

```
1. Features   build lag / rolling / seasonal features + the rain_tomorrow target
2. Train      fit the model on a time-ordered split (no leakage)
3. Track      log params, metrics, and the model to MLflow; register the best
4. Serve      load the model into a FastAPI container behind /predict
5. Monitor    score live inputs for drift (PSI / KS) and alert before accuracy rots
            └─► drift detected ──► back to step 2 (retrain on fresh data)
```

The differentiator is **step 5**. Plenty of projects stop at "trained a model";
the value here is the cheap, dependency-free early-warning system that tells you
*when the model has gone stale* — the part of MLOps that actually keeps a service
trustworthy in production.

## Architecture

```mermaid
flowchart LR
    A[Daily weather CSV] --> B[features.py<br/>lag / rolling / seasonal + target]
    B --> C{Train}
    C -->|free / CI| D[numpy LogisticRegression]
    C -->|heavy| E[sklearn / GBM]
    D --> F[MLflow tracking + registry]
    E --> F
    F --> G[FastAPI service /predict<br/>Docker]
    G --> H[Live predictions]
    H --> I[monitor: PSI / KS drift<br/>numpy core / Evidently]
    I -->|drift >= 0.2| C
    subgraph Free local
        D
        F
        G
        I
    end
    subgraph Azure ML (opt-in)
        F
        G
    end
```

The numeric core (`drift.py`, `metrics.py`, `model.py`, `features.py`) has no
dependency beyond numpy / pandas and is fully covered by **hand-derived
known-answer tests**: PSI of identical samples is ~0 and of a 3σ shift clears
0.2; the KS statistic is checked against a worked ECDF-gap example; the metrics
against tiny confusion matrices; the logistic regression reaches ~1.0 on
separable data with a monotone-decreasing loss. The MLflow / FastAPI / Evidently
/ Azure paths live behind **lazy imports** in `tracking.py`, `serve.py`, and
`monitor.py`, so the core and the test suite never pull them in.

---

## Run it

### Free / local (no cloud, no cost)

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

# 1. the reproducible demo (pure-numpy core; writes outputs/)
python -m mlpipe.cli demo

# 2. train on a real weather CSV and log the run to MLflow (local mlruns/)
mlpipe train --config config/config.yaml
mlflow ui                                  # browse runs at http://localhost:5000

# 3. serve the model behind FastAPI (locally or in the container)
docker compose up inference                # http://localhost:8000/health , /predict
#   or directly:  uvicorn mlpipe.serve:app  (with MODEL_PATH set)

# 4. run a drift report against fresh data
mlpipe monitor --reference reference.csv --current recent.csv
```

The whole local stack — MLflow tracking server + the inference service — comes up
with `docker compose up` and is validated by `docker compose config`.

### Azure ML (opt-in, **costs money**)

The same code logs to an Azure ML workspace (MLflow-compatible tracking +
registry) and serves from a managed online endpoint or Azure Container Apps. It
provisions **billable** resources and **must never be deployed without your
explicit go-ahead** — see [`azure/README.md`](azure/README.md) for the `az ml`
steps, the `endpoint.yml` skeleton, and the prominent cost warning + teardown.

---

## Model card

- **Intended use.** Short-horizon (next-day) rain-day classification for a single
  weather station, as a teaching / portfolio example of the full MLOps loop. Not
  for operational forecasting or any safety-critical decision.
- **Features.** Lagged mean temperature and precipitation (1–3 days), 3- and
  7-day rolling means of each, and a day-of-year sine/cosine seasonal encoding —
  12 features in all, every one computable from data known by the end of *today*.
- **Target.** `rain_tomorrow` = next-day precipitation ≥ 1 mm.
- **Metrics.** Reported on a **time-ordered** holdout (later days only): accuracy,
  F1 at the 0.5 threshold, and ROC-AUC. The demo headline is 0.89 / 0.67 / 0.94.
- **Drift-monitoring policy.** Per feature, compute **PSI** (quantile-binned) and
  the **KS** statistic against the training reference. PSI `< 0.1` = stable,
  `0.1–0.2` = moderate (review), `≥ 0.2` = major (investigate / retrain). The
  monitor is run on a schedule against recent inputs; a sustained major shift in
  the weather features is the retrain trigger.
- **Limitations.** Synthetic demo data; a linear model that will under-fit real
  non-linear weather; PSI thresholds are heuristics; single station and single
  operating point.

## Use your own model / data

- **Your data.** Drop a daily CSV with `date`, `tmean_c`, `precip_mm` columns at
  the path in `config/config.yaml` (the sibling
  `non-spatial/01-data-engineering-pipeline` produces exactly this shape for
  Cameroon stations) and run `mlpipe train`.
- **Your model.** `features.py` / `metrics.py` / the drift core are
  model-agnostic. Swap the numpy `LogisticRegression` for an sklearn or
  gradient-boosted model in `cli.train` (the heavy path), pickle it to
  `outputs/model.pkl`, and `mlpipe serve` / the container will load it unchanged
  as long as it exposes `predict_proba`.
- **Your drift baseline.** Point `mlpipe monitor` at any reference / current pair
  of CSVs; the report is just PSI + KS per shared column.

## Results

| Metric | Value |
| --- | --- |
| Test accuracy | 0.8939 |
| Test F1 | 0.6667 |
| Test ROC-AUC | 0.9379 |
| Features drifted (of 12) | 10 |
| Max PSI | 8.79 |

The two day-of-year features (`doy_sin`, `doy_cos`) are correctly **not** flagged
— the planted shift is meteorological, not calendrical — which is the kind of
selective signal that makes a drift report actionable rather than noise.

## Limitations

- **Synthetic baseline.** The committed numbers come from seeded synthetic data;
  they validate the pipeline, not real-world skill.
- **Leakage risk.** Features are built with a strict no-look-ahead shift and the
  split is time-ordered, but any real deployment must re-audit that the serving
  features use only information available at prediction time.
- **Drift thresholds are heuristics.** PSI 0.1 / 0.2 are conventions; calibrate
  them to your own false-alarm tolerance.
- **Serverless cold starts.** A scale-to-zero container (Azure Container Apps) is
  cheap but adds first-request latency; a managed online endpoint avoids the cold
  start but bills continuously.

## License

MIT © 2026 Joseph Mbuh
