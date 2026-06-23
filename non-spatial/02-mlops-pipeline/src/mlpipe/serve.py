"""FastAPI inference service behind a lazy import.

This module turns the trained rain-day model into an HTTP service. It is the
"serving" leg of the loop: a container exposes two endpoints —

* ``GET  /health``  — liveness probe (returns ``{"status": "ok"}``);
* ``POST /predict``  — body is a JSON object of the engineered features (the keys
  in :data:`mlpipe.features.FEATURE_COLUMNS`); the response is the rain
  probability and the 0/1 decision.

Run it locally with uvicorn::

    export MODEL_PATH=outputs/model.pkl
    uvicorn mlpipe.serve:app --host 0.0.0.0 --port 8000

or inside the container built from the project ``Dockerfile`` (``docker compose
up inference``). ``fastapi`` / ``pydantic`` / ``uvicorn`` are imported lazily, so
importing this module costs nothing and the test suite never pulls them in.

``app`` at module scope is created on first attribute access by uvicorn through
:func:`create_app`, reading ``MODEL_PATH`` from the environment.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlpipe.features import FEATURE_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def _load_model(model_path: str | Path) -> Any:
    """Unpickle the fitted model from disk (raises if absent)."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file {path} not found. Train first (mlpipe train) or set "
            "MODEL_PATH to a pickled model."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def create_app(model_path: str | Path) -> Any:
    """Build and return a FastAPI app serving the model at ``model_path``.

    The app loads the pickled model once at startup and exposes ``/health`` and
    ``/predict``. ``fastapi`` and ``numpy`` are imported here so importing the
    module stays cheap.
    """
    import numpy as np
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    model = _load_model(model_path)

    class Features(BaseModel):
        """One day's engineered features (see mlpipe.features.FEATURE_COLUMNS)."""

        tmean_lag1: float = Field(..., description="Mean temp 1 day ago (C).")
        tmean_lag2: float
        tmean_lag3: float
        precip_lag1: float = Field(..., description="Precip 1 day ago (mm).")
        precip_lag2: float
        precip_lag3: float
        tmean_roll3: float
        tmean_roll7: float
        precip_roll3: float
        precip_roll7: float
        doy_sin: float
        doy_cos: float

    app = FastAPI(title="Rain-day prediction", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(features: Features) -> dict[str, Any]:
        row = np.array([[getattr(features, name) for name in FEATURE_COLUMNS]])
        proba = float(model.predict_proba(row)[0])
        return {
            "rain_probability": proba,
            "rain_tomorrow": int(proba >= 0.5),
            "threshold": 0.5,
        }

    return app


def _build_default_app() -> Any:
    """Lazy app factory for ``uvicorn mlpipe.serve:app`` (reads ``MODEL_PATH``)."""
    return create_app(os.environ.get("MODEL_PATH", "outputs/model.pkl"))


def __getattr__(name: str) -> Any:
    # Defer app construction (and the FastAPI import) until uvicorn asks for it.
    if name == "app":
        return _build_default_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
