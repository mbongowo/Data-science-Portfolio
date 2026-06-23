"""Experiment tracking behind a lazy MLflow import.

This module closes the "notebook to tracked experiment" gap. It is a thin
wrapper over MLflow so that training runs — parameters, metrics, and the fitted
model — are logged, comparable, and registerable, instead of living in a
forgotten notebook cell.

Two tracking backends, same code:

* **Free / local** — set ``MLFLOW_TRACKING_URI`` to a local path (or leave it
  unset to use the default ``mlruns/`` folder) and browse runs with ``mlflow
  ui``. Nothing leaves your machine and nothing costs money.
* **Azure ML (opt-in, costs money)** — point ``MLFLOW_TRACKING_URI`` at an Azure
  Machine Learning workspace's MLflow tracking URI (see ``azure/README.md``).
  The same calls then log to the cloud workspace and the model can be promoted to
  the Azure ML model registry / a managed online endpoint.

``mlflow`` is imported lazily inside each function, so importing this module is
cheap and the test suite never pulls it in.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator


def _configure_tracking_uri() -> str:
    """Resolve and set the active MLflow tracking URI; return it."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(uri)
    return uri


@contextmanager
def start_run(
    experiment: str = "rain-day", run_name: str | None = None
) -> Iterator[Any]:
    """Context manager that opens an MLflow run under ``experiment``.

    Usage::

        with start_run(run_name="numpy-baseline") as run:
            log_training(params, metrics, model)

    The tracking URI comes from ``MLFLOW_TRACKING_URI`` (local ``mlruns/`` by
    default, or an Azure ML workspace URI for the cloud path).
    """
    import mlflow

    _configure_tracking_uri()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_training(
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any | None = None,
    artifact_path: str = "model",
) -> None:
    """Log hyperparameters, metrics, and (optionally) the fitted model.

    Must be called inside a :func:`start_run` context. The model is pickled and
    logged as an artifact so a later ``register_best`` (or the serving container)
    can load exactly this object.
    """
    import pickle
    import tempfile
    from pathlib import Path

    import mlflow

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if model is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            with open(path, "wb") as fh:
                pickle.dump(model, fh)
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def register_best(
    experiment: str = "rain-day",
    metric: str = "test_roc_auc",
    model_name: str = "rain-day-classifier",
) -> Any:
    """Register the best run's model in the MLflow model registry.

    Searches the experiment for the run with the highest ``metric`` and registers
    its logged model under ``model_name``. On the Azure ML backend this promotes
    the model into the workspace registry, ready for a managed online endpoint.
    """
    import mlflow

    _configure_tracking_uri()
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"Experiment {experiment!r} not found.")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment {experiment!r}.")
    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model/model.pkl"
    return mlflow.register_model(model_uri, model_name)
