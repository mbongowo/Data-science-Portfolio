"""Data layer: load the crop table, encode labels, build X, and split.

Pure pandas/numpy. The dataset is a wide table with the seven agronomic feature
columns and a string crop label:

    N, P, K, temperature, humidity, ph, rainfall, label

This matches the public Kaggle *Crop Recommendation* schema, so the same code
reads either the bundled synthetic CSV or the real dataset unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The seven model features, in canonical order.
FEATURE_COLUMNS: list[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
LABEL_COLUMN: str = "label"


def load_crops(csv_path: str | Path) -> pd.DataFrame:
    """Load the crop CSV and validate that the expected columns are present.

    Returns the DataFrame with the feature columns coerced to float and the
    label column to string, ordered as ``FEATURE_COLUMNS + [LABEL_COLUMN]``.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    out = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float)
    out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(str)
    return out.reset_index(drop=True)


def encode_labels(labels) -> tuple[np.ndarray, np.ndarray]:
    """Map string crop labels to integer indices.

    Returns ``(y_int, classes)`` where ``classes`` is the sorted array of unique
    labels and ``y_int[i]`` is the index of ``labels[i]`` in ``classes``. The
    sorted order makes the encoding deterministic and round-trippable:
    ``classes[y_int] == labels``.
    """
    labels = np.asarray(labels, dtype=object)
    classes = np.array(sorted(set(labels.tolist())), dtype=object)
    index = {c: i for i, c in enumerate(classes)}
    y_int = np.array([index[c] for c in labels], dtype=int)
    return y_int, classes


def feature_matrix(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> np.ndarray:
    """Return the feature columns of ``df`` as a float ``ndarray``.

    Defaults to :data:`FEATURE_COLUMNS`; raises if any requested column is
    missing so a typo fails loudly rather than silently dropping a feature.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not in DataFrame: {missing}")
    return df[cols].to_numpy(dtype=float)


def stratified_split(
    y: np.ndarray, fractions: tuple[float, ...], seed: int = 0
) -> list[np.ndarray]:
    """Split row indices into stratified parts honouring ``fractions``.

    Each class is shuffled (seeded) and partitioned by the cumulative
    fractions, so every part keeps roughly each class's share and the parts are
    disjoint and cover all rows. ``fractions`` must be positive and sum to ~1.
    Returns a list of index arrays, one per fraction.
    """
    fractions = tuple(float(f) for f in fractions)
    if any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")

    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in fractions]

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = idx.shape[0]
        # Cumulative cut points; the last part takes the remainder so every row
        # is assigned exactly once.
        cuts = np.cumsum([int(round(f * n)) for f in fractions[:-1]])
        chunks = np.split(idx, cuts)
        for part, chunk in zip(parts, chunks, strict=True):
            part.extend(chunk.tolist())

    return [np.array(sorted(p), dtype=int) for p in parts]
