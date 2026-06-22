"""Runnable, GPU-free demo of the pure-numpy classification stack.

``run_demo`` deterministically synthesises a small EuroSAT-like dataset — a set
of multispectral patches, one per sample, with ``K`` land-cover classes that
each carry a distinct per-band spectral signature plus Gaussian noise — then
runs the *real* core end to end: featurise each patch with
:func:`lcnet.data.patch_features`, split with a stratified train/val/test split,
standardize on the training statistics, train the from-scratch
:class:`lcnet.classifier.SoftmaxClassifier`, and evaluate on the held-out test
split with the :mod:`lcnet.metrics` suite.

Nothing here touches torch, torchgeo, or real imagery: it exists so a reader can
clone the repo and see honest numbers fall out of the tested code in well under
two seconds. The signal-to-noise ratio is tuned so accuracy is strong but
realistic (mid-0.8s to mid-0.9s), not a trivial 1.0. Everything is seeded, so a
given seed always yields the same numbers.

Run it::

    python -m lcnet.demo            # writes outputs/ and prints the summary
    python -m lcnet.cli demo        # same, via the CLI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lcnet.classifier import SoftmaxClassifier, standardize
from lcnet.data import patch_features, stratified_split
from lcnet.metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    micro_f1,
    per_class_f1,
    top_k_accuracy,
)

__all__ = ["run_demo", "main"]

# Scene / dataset geometry. Small enough to run in well under two seconds.
N_PER_CLASS = 80
N_CLASSES = 6
N_BANDS = 6
PATCH_HW = 8  # each patch is (N_BANDS, 8, 8)

# Per-class, per-band mean signatures (rows = classes, cols = bands). These are
# stylised reflectance-like profiles: vegetation bright in the NIR-ish bands,
# water dark, built-up flat-and-bright, etc. Several classes share a similar
# overall brightness and differ only in band *shape* (forest vs crop, built-up
# vs bare soil), so they overlap once per-sample noise is added.
_SIGNATURES = np.array(
    [
        [0.30, 0.34, 0.40, 0.62, 0.50, 0.36],  # forest
        [0.33, 0.37, 0.46, 0.56, 0.48, 0.38],  # crop / herbaceous
        [0.16, 0.18, 0.17, 0.15, 0.13, 0.12],  # water
        [0.46, 0.48, 0.49, 0.50, 0.51, 0.52],  # built-up / urban
        [0.50, 0.52, 0.53, 0.48, 0.44, 0.42],  # bare soil
        [0.34, 0.42, 0.52, 0.46, 0.40, 0.34],  # wetland / mixed
    ],
    dtype=np.float64,
)

# Per-sample additive shift applied to the WHOLE patch (a constant per band), on
# top of pixel-level noise. Because patch_features averages each band over the
# patch, this per-sample shift — not the pixel noise, which averages out — is
# what keeps the per-band mean feature noisy and the accuracy realistic.
SAMPLE_STD = 0.045
PIXEL_STD = 0.06


def _synthesize(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X_features, y)`` from synthetic multispectral patches.

    For each sample a ``(N_BANDS, PATCH_HW, PATCH_HW)`` patch is drawn around its
    class signature with Gaussian noise, then reduced to a per-band [mean, std]
    feature vector — the same featurisation the real pipeline would apply to
    EuroSAT patches.
    """
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for cls in range(N_CLASSES):
        sig = _SIGNATURES[cls]
        for _ in range(N_PER_CLASS):
            # Per-sample per-band shift (survives the patch mean) + pixel noise
            # (averages out). The shift is what keeps the task non-trivial.
            shift = rng.normal(0.0, SAMPLE_STD, size=N_BANDS)
            patch = (sig + shift)[:, None, None] + rng.normal(
                0.0, PIXEL_STD, size=(N_BANDS, PATCH_HW, PATCH_HW)
            )
            patch = np.clip(patch, 0.0, 1.0)
            feats.append(patch_features(patch))
            labels.append(cls)
    X = np.vstack(feats)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def run_demo(seed: int = 0, out_dir: str = "outputs") -> dict:
    """Run the classification demo and write artifacts.

    Parameters
    ----------
    seed : int, optional
        Seed for the synthetic data and the classifier init. Fixed seed -> fixed
        numbers.
    out_dir : str, optional
        Directory for ``confusion_matrix.csv`` and ``metrics.json``. Created if
        missing.

    Returns
    -------
    dict
        ``n_samples``, ``n_classes``, ``n_features``, ``test_accuracy``,
        ``test_macro_f1``, ``per_class_f1`` (list).
    """
    rng = np.random.default_rng(seed)
    X, y = _synthesize(rng)
    n_samples, n_features = X.shape

    # Deterministic, class-balanced split.
    train_idx, val_idx, test_idx = stratified_split(y, (0.6, 0.2, 0.2), seed=seed)

    # Standardize on TRAIN statistics; apply the same scaling to val/test.
    X_train, mean, std = standardize(X[train_idx])
    X_val, _, _ = standardize(X[val_idx], mean, std)
    X_test, _, _ = standardize(X[test_idx], mean, std)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Train the from-scratch softmax baseline.
    clf = SoftmaxClassifier(num_classes=N_CLASSES)
    clf.fit(X_train, y_train, lr=0.5, epochs=400, l2=1e-3, seed=seed)

    # Evaluate on the held-out test split.
    proba = clf.predict_proba(X_test)
    y_pred = np.argmax(proba, axis=1)

    test_acc = accuracy(y_test, y_pred)
    test_macro = macro_f1(y_test, y_pred, N_CLASSES)
    test_micro = micro_f1(y_test, y_pred, N_CLASSES)
    pcf1 = per_class_f1(y_test, y_pred, N_CLASSES)
    kappa = cohen_kappa(y_test, y_pred, N_CLASSES)
    top2 = top_k_accuracy(y_test, proba, k=2)
    val_acc = accuracy(y_val, clf.predict(X_val))
    cm = confusion_matrix(y_test, y_pred, N_CLASSES)

    # --- artifacts ----------------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    header = "true\\pred," + ",".join(str(k) for k in range(N_CLASSES))
    cm_lines = [header]
    for i in range(N_CLASSES):
        cm_lines.append(str(i) + "," + ",".join(str(int(v)) for v in cm[i]))
    (out_path / "confusion_matrix.csv").write_text("\n".join(cm_lines) + "\n")

    metrics = {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_classes": int(N_CLASSES),
        "n_features": int(n_features),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro),
        "test_micro_f1": float(test_micro),
        "test_cohen_kappa": float(kappa),
        "test_top2_accuracy": float(top2),
        "per_class_f1": [float(v) for v in pcf1],
        "final_train_loss": float(clf.loss_history[-1]),
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    return {
        "n_samples": int(n_samples),
        "n_classes": int(N_CLASSES),
        "n_features": int(n_features),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro),
        "per_class_f1": [float(v) for v in pcf1],
    }


def main(argv: list[str] | None = None) -> None:
    """Console entry point for ``python -m lcnet.demo``."""
    parser = argparse.ArgumentParser(description="Run the lcnet classification demo.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic-data seed.")
    parser.add_argument(
        "--out-dir", default="outputs", help="Where to write artifacts."
    )
    args = parser.parse_args(argv)

    result = run_demo(seed=args.seed, out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    print(f"\nArtifacts written to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
