# MLOps Pipeline — rain-day classification with drift monitoring

The MLOps Pipeline trains a logistic-regression classifier that predicts whether
it will rain on Cameroon weather data, and wraps it in production monitoring. A
pure-numpy core detects feature drift using the Population Stability Index (PSI)
and the Kolmogorov-Smirnov (KS) test, so shifts in the input distribution are
caught before they degrade predictions. MLflow tracks experiments and a
containerised FastAPI service serves the model, with Evidently dashboards for
monitoring.

There is a free local path built on DuckDB and MLflow, plus an opt-in Azure ML
path. The project shows the full lifecycle: train, serve, monitor and detect
drift.
