"""One-command, reproducible demo of the pure-numpy crop recommender.

This synthesises a **seeded** crop dataset for ten Cameroon-relevant crops —
maize, cassava, plantain, cocoa, groundnut, rice, sorghum, yam, beans and
oil_palm — by giving each crop a distinct centre in the seven-feature space
(realistic N/P/K, temperature, humidity, pH and rainfall ranges) and sampling
around it with Gaussian noise. It then runs the *real* core: a stratified split,
:func:`~croprec.model.standardize` fitted on the training rows, the numpy
:class:`~croprec.model.SoftmaxClassifier`, and the
:mod:`~croprec.metrics` on the held-out test split.

The signal-to-noise is tuned so the classifier reaches a strong-but-realistic
test accuracy (high-0.8s to mid-0.9s), not a suspicious 1.0. The metrics are
deterministic and pinned by a test, so the README numbers stay honest. Only
numpy/pandas + stdlib are required, so it runs anywhere including CI.

Run it with ``python -m croprec.cli demo`` or ``run_demo(seed=0)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from croprec.data import FEATURE_COLUMNS, encode_labels, stratified_split
from croprec.metrics import accuracy, confusion_matrix, macro_f1, top_k_accuracy
from croprec.model import SoftmaxClassifier, standardize

# Per-crop feature centres (N, P, K, temperature C, humidity %, pH, rainfall mm).
# Values are plausible agronomic ranges, chosen so the ten crops occupy
# reasonably separated — but overlapping — regions of the feature space.
CROP_CENTERS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "maize":     (80, 45, 40, 24, 65, 6.2, 90),
    "cassava":   (40, 30, 35, 27, 60, 5.8, 110),
    "plantain":  (70, 50, 70, 26, 78, 6.0, 160),
    "cocoa":     (55, 35, 50, 25, 85, 6.5, 180),
    "groundnut": (30, 55, 45, 28, 55, 6.4, 70),
    "rice":      (90, 50, 40, 24, 82, 6.0, 220),
    "sorghum":   (60, 40, 35, 29, 45, 6.8, 55),
    "yam":       (50, 45, 60, 27, 70, 6.1, 130),
    "beans":     (35, 65, 50, 22, 60, 6.6, 100),
    "oil_palm":  (65, 30, 55, 27, 88, 5.0, 200),
}

# Per-feature noise (std). Larger than a clean separation so the problem is
# realistic and the classifier lands below 1.0 accuracy.
_FEATURE_STD = np.array([11.0, 9.0, 9.0, 1.8, 7.0, 0.38, 22.0])

_SAMPLES_PER_CROP = 80
_SPLIT_FRACTIONS = (0.7, 0.3)


def _synthesize(seed: int) -> pd.DataFrame:
    """Deterministically synthesise the crop dataset.

    Returns a DataFrame with the seven feature columns plus ``label``. Features
    are clipped to non-negative; humidity to 0-100 and pH to 3.5-9.0 so the
    synthetic rows stay physically plausible.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for crop, center in CROP_CENTERS.items():
        center_arr = np.array(center, dtype=float)
        noise = rng.normal(0.0, 1.0, size=(_SAMPLES_PER_CROP, 7)) * _FEATURE_STD
        rows = center_arr + noise
        rows = np.clip(rows, 0.0, None)
        rows[:, 4] = np.clip(rows[:, 4], 0.0, 100.0)  # humidity %
        rows[:, 5] = np.clip(rows[:, 5], 3.5, 9.0)    # pH
        frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        frame["label"] = crop
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    # Shuffle the combined frame so rows are not grouped by crop.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the end-to-end demo: synthesise, train, evaluate, write artifacts.

    Parameters
    ----------
    seed:
        Seed for the synthetic data, the split and the classifier init, so the
        returned metrics are deterministic.
    out_dir:
        Directory for ``confusion_matrix.csv`` and ``metrics.json`` (created if
        missing).

    Returns
    -------
    dict
        ``n_samples``, ``n_crops``, ``n_features``, ``test_accuracy``,
        ``test_macro_f1`` and ``test_top3_accuracy``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _synthesize(seed)
    y, classes = encode_labels(df["label"].to_numpy())
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    n_crops = int(classes.shape[0])

    train_idx, test_idx = stratified_split(y, _SPLIT_FRACTIONS, seed=seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit the z-score scaler on the training rows; reuse it for the test rows.
    X_train_s, mean, std = standardize(X_train)
    X_test_s, _, _ = standardize(X_test, mean=mean, std=std)

    model = SoftmaxClassifier()
    model.fit(X_train_s, y_train, lr=0.5, epochs=600, l2=1e-3, seed=seed)

    proba = model.predict_proba(X_test_s)
    y_pred = model.predict(X_test_s)

    test_accuracy = accuracy(y_test, y_pred)
    test_macro_f1 = macro_f1(y_test, y_pred, n_crops)
    test_top3_accuracy = top_k_accuracy(y_test, proba, k=3)

    cm = confusion_matrix(y_test, y_pred, n_crops)

    metrics = {
        "seed": int(seed),
        "n_samples": int(X.shape[0]),
        "n_crops": n_crops,
        "n_features": int(X.shape[1]),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_macro_f1": round(float(test_macro_f1), 4),
        "test_top3_accuracy": round(float(test_top3_accuracy), 4),
    }

    # --- Write artifacts ------------------------------------------------------
    cm_df = pd.DataFrame(cm, index=list(classes), columns=list(classes))
    cm_df.to_csv(out_path / "confusion_matrix.csv")
    with (out_path / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


if __name__ == "__main__":  # pragma: no cover - manual entry point
    print(json.dumps(run_demo(0), indent=2))
