"""Optional stronger model: a scikit-learn RandomForest on the real dataset.

This is the documented upgrade path, **not** the CI-tested core and **not**
imported by the test suite. The deployed app uses the pure-numpy
:class:`~croprec.model.SoftmaxClassifier` it trains itself, so nothing here is
needed to run the app.

To use it, download the public *Crop Recommendation* dataset from Kaggle
(``atharvaingle/crop-recommendation-dataset``) — a CSV with the seven feature
columns ``N, P, K, temperature, humidity, ph, rainfall`` and a ``label`` column —
then::

    pip install scikit-learn joblib
    python -c "from croprec.train import train_random_forest; \\
               print(train_random_forest('Crop_recommendation.csv'))"

scikit-learn and joblib are imported lazily inside the function so importing this
module stays cheap and free of the heavy dependency.
"""

from __future__ import annotations

from pathlib import Path

from croprec.data import FEATURE_COLUMNS, encode_labels, load_crops, stratified_split


def train_random_forest(
    csv_path: str | Path,
    model_out: str | Path = "models/crop_rf.joblib",
    n_estimators: int = 300,
    seed: int = 0,
) -> dict:
    """Train a RandomForest on the real Kaggle dataset and save it.

    Imports scikit-learn and joblib lazily. Returns a metrics dict
    (``test_accuracy``, ``test_macro_f1``, ``n_classes``, ``model_path``) and
    writes the fitted model to ``model_out``.
    """
    import joblib  # noqa: PLC0415 - lazy, optional dependency
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    from sklearn.metrics import accuracy_score, f1_score  # noqa: PLC0415

    df = load_crops(csv_path)
    y, classes = encode_labels(df["label"].to_numpy())
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    train_idx, test_idx = stratified_split(y, (0.8, 0.2), seed=seed)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X[train_idx], y[train_idx])

    y_pred = clf.predict(X[test_idx])
    test_accuracy = float(accuracy_score(y[test_idx], y_pred))
    test_macro_f1 = float(f1_score(y[test_idx], y_pred, average="macro"))

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": classes}, out)

    return {
        "test_accuracy": round(test_accuracy, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "n_classes": int(classes.shape[0]),
        "model_path": str(out),
    }
