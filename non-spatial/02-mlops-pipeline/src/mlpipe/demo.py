"""One-command, reproducible demo of the pure-numpy MLOps core.

This drives the *real* numeric core end-to-end on seeded synthetic weather data,
with nothing beyond numpy / pandas + the standard library, so it runs anywhere
(including CI) in well under two seconds. No MLflow, FastAPI, or Evidently is
touched.

What it does, using the same functions the CLI and service use:

1. **Synthesize a reference regime.** A seeded daily weather series for a single
   Cameroon station whose precipitation depends on temperature and a seasonal
   term, so tomorrow's rain is genuinely (noisily) predictable from recent
   weather — not random.
2. **Train and evaluate.** Build features with the real
   :func:`mlpipe.features.make_features`, split in time order with
   :func:`~mlpipe.features.train_test_split_time`, fit the pure-numpy
   :class:`mlpipe.model.LogisticRegression` on the train split, and score the
   later holdout with the real metrics (accuracy, F1, ROC-AUC).
3. **Plant a distribution shift and detect it.** Synthesize a *current* regime
   that is warmer and wetter than the reference, then run
   :func:`mlpipe.drift.feature_drift_report` reference-vs-current. The planted
   shift lands squarely on the temperature and precipitation features, and the
   PSI/KS report flags exactly those — the honest, reproducible headline that a
   live model would be scoring out-of-distribution inputs.

Artifacts ``outputs/metrics.json`` and ``outputs/drift_report.csv`` are written,
and a summary dict is returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mlpipe.drift import feature_drift_report
from mlpipe.features import FEATURE_COLUMNS, make_features, train_test_split_time
from mlpipe.metrics import accuracy, f1, roc_auc
from mlpipe.model import LogisticRegression, standardize

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# Demo constants. Chosen so rain is genuinely predictable (accuracy ~0.7-0.9),
# the run stays under a couple of seconds, and the planted shift is unambiguous.
N_DAYS = 900
START_DATE = "2021-01-01"

# Logistic-regression hyperparameters for the rain-day classifier.
LR = 0.3
EPOCHS = 800
L2 = 0.001
MODEL_SEED = 0

TRAIN_FRAC = 0.8


def synthesize_weather(
    seed: int = 0,
    *,
    temp_offset: float = 0.0,
    wet_boost: float = 0.0,
) -> pd.DataFrame:
    """Synthesize a seeded daily weather series for one station.

    Daily mean temperature follows a seasonal cycle plus noise. Precipitation is
    driven by a latent "rain propensity" that rises with warmth and a seasonal
    wet-season term, passed through a noisy threshold — so whether it rains
    tomorrow really does depend on recent weather, which is what makes the
    classifier better than chance.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng`` (reproducible bit-for-bit).
    temp_offset:
        Degrees added to every day's temperature. Used to plant a warm shift in
        the *current* regime.
    wet_boost:
        Added to the rain propensity, making rain more frequent and heavier. Used
        to plant a wet shift in the *current* regime.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``tmean_c``, ``precip_mm``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    doy = dates.dayofyear.to_numpy(dtype=float)

    # Seasonal temperature near a tropical station, plus offset and AR(1)-ish noise.
    season = 3.0 * np.sin(2.0 * np.pi * (doy - 80.0) / 365.25)
    tmean = 26.0 + temp_offset + season + rng.normal(0.0, 1.2, size=N_DAYS)

    # Wet-season seasonal term (peaks mid-year) feeds a rain propensity that also
    # grows with warmth; precipitation is propensity-driven with many dry days.
    wet_season = 2.0 * np.sin(2.0 * np.pi * (doy - 150.0) / 365.25)
    propensity = (
        -0.2
        + 0.25 * (tmean - 26.0)
        + wet_season
        + wet_boost
        + rng.normal(0.0, 0.8, size=N_DAYS)
    )
    rains = propensity > 0.0
    amount = np.where(rains, rng.gamma(2.0, 3.0, size=N_DAYS) + wet_boost, 0.0)
    precip = np.clip(amount, 0.0, None)

    return pd.DataFrame(
        {"date": dates, "tmean_c": np.round(tmean, 2), "precip_mm": np.round(precip, 2)}
    )


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Run the full synthetic demo end-to-end and write artifacts.

    Parameters
    ----------
    seed:
        Seed threaded through the reference data synthesis. The current-regime
        seed and the model seed are fixed so the headline numbers are stable.
    out_dir:
        Directory for the artifacts (created if missing): ``metrics.json`` and
        ``drift_report.csv``.

    Returns
    -------
    dict
        ``n_train``, ``test_accuracy``, ``test_f1``, ``test_roc_auc``,
        ``n_features_drifted``, ``max_psi``, and ``drifted_features`` (list).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Reference regime: train and evaluate the rain-day model ----------
    reference_raw = synthesize_weather(seed)
    feats = make_features(reference_raw)
    train_df, test_df = train_test_split_time(feats, frac=TRAIN_FRAC)

    X_train = train_df[FEATURE_COLUMNS].to_numpy()
    y_train = train_df["rain_tomorrow"].to_numpy()
    X_test = test_df[FEATURE_COLUMNS].to_numpy()
    y_test = test_df["rain_tomorrow"].to_numpy()

    # Standardize on train statistics, reuse them on test (no leakage).
    X_train_s, mu, sd = standardize(X_train)
    X_test_s, _, _ = standardize(X_test, mean=mu, std=sd)

    model = LogisticRegression(lr=LR, epochs=EPOCHS, l2=L2, seed=MODEL_SEED)
    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)
    preds = (proba >= 0.5).astype(int)
    test_accuracy = round(accuracy(y_test, preds), 4)
    test_f1 = round(f1(y_test, preds), 4)
    test_roc_auc = round(roc_auc(y_test, proba), 4)

    # --- Current regime: plant a warmer, wetter shift and detect it -------
    current_raw = synthesize_weather(seed + 1, temp_offset=3.5, wet_boost=2.5)
    current_feats = make_features(current_raw)

    ref_feature_frame = feats[FEATURE_COLUMNS]
    cur_feature_frame = current_feats[FEATURE_COLUMNS]
    drift = feature_drift_report(ref_feature_frame, cur_feature_frame)
    summary = drift.attrs["summary"]

    max_psi = round(float(drift["psi"].max()), 4) if len(drift) else 0.0

    result: dict[str, Any] = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "n_features_drifted": int(summary["n_drifted"]),
        "max_psi": max_psi,
        "drifted_features": list(summary["drifted_features"]),
    }

    # --- Artifacts --------------------------------------------------------
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    drift.to_csv(out / "drift_report.csv", index=False)

    return result


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_demo(0), indent=2))
