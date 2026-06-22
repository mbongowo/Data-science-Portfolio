"""Pure-numpy segmentation metrics.

These functions depend only on :mod:`numpy`, so they import and test without
torch/lightning installed. They cover two regimes:

* Binary masks (foreground vs. background), via :func:`iou_score`,
  :func:`f1_score`, :func:`precision_score`, :func:`recall_score`, and
  :func:`confusion_counts`. Inputs may be booleans, {0,1} integers, or
  probability maps (thresholded with ``threshold``).
* Multi-class integer label maps, via :func:`per_class_iou` and
  :func:`mean_iou_multiclass`. Inputs are integer arrays of class indices in
  ``range(num_classes)``.

Both regimes support ``ignore_index``: pixels whose *target* equals
``ignore_index`` are excluded from every count. This matches the convention
used by torch losses for unlabelled / void pixels.

Empty-mask convention
----------------------
For the binary scores, when the union (or the relevant denominator) is zero the
prediction and target agree that nothing is present, so the score is ``1.0``.
For per-class IoU a class that is absent from both prediction and target after
masking is reported as ``nan``; :func:`mean_iou_multiclass` skips ``nan``
classes when averaging so absent classes do not drag the mean toward zero.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "iou_score",
    "f1_score",
    "precision_score",
    "recall_score",
    "confusion_counts",
    "mean_iou",
    "per_class_iou",
    "mean_iou_multiclass",
    "confusion_matrix",
    "pixel_accuracy",
    "frequency_weighted_iou",
    "cohen_kappa",
    "per_class_precision",
    "per_class_recall",
]


def _binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Coerce probabilities / ints / bools into a boolean mask.

    Parameters
    ----------
    arr : numpy.ndarray
        Array of probabilities, integers, or booleans.
    threshold : float
        Values ``>= threshold`` become ``True``. Ignored for boolean input.

    Returns
    -------
    numpy.ndarray
        Boolean array with the same shape as ``arr``.
    """
    a = np.asarray(arr)
    if a.dtype == bool:
        return a
    return a >= threshold


def _valid_mask(
    target: np.ndarray, ignore_index: int | None
) -> np.ndarray | None:
    """Return a boolean keep-mask excluding ``ignore_index`` pixels, or None.

    Parameters
    ----------
    target : numpy.ndarray
        Target array (before binarisation).
    ignore_index : int or None
        Target value to drop. If ``None``, all pixels are kept and the function
        returns ``None`` so callers can skip masking entirely.

    Returns
    -------
    numpy.ndarray or None
        Boolean array that is ``True`` for pixels to keep, or ``None`` when no
        masking is requested.
    """
    if ignore_index is None:
        return None
    return np.asarray(target) != ignore_index


def confusion_counts(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` for binary masks.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Probabilities, ints, or bools of identical shape.
    threshold : float, optional
        Threshold applied to non-boolean inputs.
    ignore_index : int, optional
        Pixels whose ``target`` equals this value are excluded from all four
        counts. Applied before binarisation, so it sees the raw target labels.

    Returns
    -------
    tuple of int
        ``(true_positive, false_positive, false_negative, true_negative)``.

    Raises
    ------
    ValueError
        If ``pred`` and ``target`` have different shapes.
    """
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"shape mismatch: pred {pred_arr.shape} vs target {target_arr.shape}"
        )
    keep = _valid_mask(target_arr, ignore_index)
    p = _binarize(pred_arr, threshold)
    t = _binarize(target_arr, threshold)
    if keep is not None:
        p = p[keep]
        t = t[keep]
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, np.logical_not(t)).sum())
    fn = int(np.logical_and(np.logical_not(p), t).sum())
    tn = int(np.logical_and(np.logical_not(p), np.logical_not(t)).sum())
    return tp, fp, fn, tn


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> float:
    """Intersection-over-Union (Jaccard index) for binary masks.

    ``IoU = |A and B| / |A or B| = TP / (TP + FP + FN)``.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Binary masks or probabilities of identical shape.
    threshold : float, optional
        Threshold applied to non-boolean inputs.
    ignore_index : int, optional
        Target value to exclude from the counts.

    Returns
    -------
    float
        IoU in ``[0, 1]``. If the union is zero (both masks empty after
        masking) the score is ``1.0``; a perfect non-empty match is exactly
        ``1.0`` (no epsilon offset, since the denominator is already guarded).
    """
    tp, fp, fn, _ = confusion_counts(pred, target, threshold, ignore_index)
    union = tp + fp + fn
    if union == 0:
        return 1.0
    return tp / union


def f1_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> float:
    """Dice / F1 score for binary masks.

    ``F1 = 2*TP / (2*TP + FP + FN)``.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Binary masks or probabilities of identical shape.
    threshold : float, optional
        Threshold applied to non-boolean inputs.
    ignore_index : int, optional
        Target value to exclude from the counts.

    Returns
    -------
    float
        F1 in ``[0, 1]``. If the denominator is zero (both masks empty after
        masking) the score is ``1.0``.
    """
    tp, fp, fn, _ = confusion_counts(pred, target, threshold, ignore_index)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp) / denom


def precision_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> float:
    """Precision (positive predictive value) for binary masks.

    ``precision = TP / (TP + FP)``.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Binary masks or probabilities of identical shape.
    threshold : float, optional
        Threshold applied to non-boolean inputs.
    ignore_index : int, optional
        Target value to exclude from the counts.

    Returns
    -------
    float
        Precision in ``[0, 1]``. If nothing is predicted positive
        (``TP + FP == 0``) the score is ``1.0``: there are no false alarms.
    """
    tp, fp, _, _ = confusion_counts(pred, target, threshold, ignore_index)
    denom = tp + fp
    if denom == 0:
        return 1.0
    return tp / denom


def recall_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> float:
    """Recall (sensitivity, true positive rate) for binary masks.

    ``recall = TP / (TP + FN)``.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Binary masks or probabilities of identical shape.
    threshold : float, optional
        Threshold applied to non-boolean inputs.
    ignore_index : int, optional
        Target value to exclude from the counts.

    Returns
    -------
    float
        Recall in ``[0, 1]``. If the target has no positives
        (``TP + FN == 0``) the score is ``1.0``: nothing was missed.
    """
    tp, _, fn, _ = confusion_counts(pred, target, threshold, ignore_index)
    denom = tp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def mean_iou(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> float:
    """Mean binary IoU over a batch.

    ``preds`` and ``targets`` are stacked along axis 0; binary IoU is computed
    per item and averaged.

    Parameters
    ----------
    preds, targets : numpy.ndarray
        Batched masks or probabilities of identical shape, with a leading batch
        dimension.
    threshold : float, optional
        Threshold applied to non-boolean inputs.
    ignore_index : int, optional
        Target value to exclude from each item's counts.

    Returns
    -------
    float
        Mean per-item IoU. An empty batch returns ``1.0``.

    Raises
    ------
    ValueError
        If shapes differ or there is no batch dimension.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )
    if preds.ndim == 0:
        raise ValueError("expected a batch dimension")
    scores = [
        iou_score(p, t, threshold, ignore_index=ignore_index)
        for p, t in zip(preds, targets, strict=True)
    ]
    return float(np.mean(scores)) if scores else 1.0


def per_class_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    """IoU for each class of an integer label map.

    For class ``k`` the prediction and target are reduced to the boolean masks
    ``pred == k`` and ``target == k``, and IoU is ``TP / (TP + FP + FN)``.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Integer arrays of class indices, identical shape.
    num_classes : int
        Number of classes. Output has this length, indexed by class id.
    ignore_index : int, optional
        Pixels whose ``target`` equals this value are dropped before any class
        is scored.

    Returns
    -------
    numpy.ndarray
        Float array of length ``num_classes``. A class absent from both the
        (masked) prediction and target has an empty union and is reported as
        ``nan`` so it can be skipped when averaging.

    Raises
    ------
    ValueError
        If shapes differ or ``num_classes`` is not positive.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    keep = _valid_mask(target, ignore_index)
    if keep is not None:
        pred = pred[keep]
        target = target[keep]
    ious = np.empty(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p = pred == k
        t = target == k
        inter = int(np.logical_and(p, t).sum())
        union = int(np.logical_or(p, t).sum())
        if union == 0:
            ious[k] = np.nan
        else:
            ious[k] = inter / union
    return ious


def mean_iou_multiclass(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Mean IoU over classes of an integer label map (macro IoU).

    Computes :func:`per_class_iou` and averages the classes that are present
    (non-``nan``). Absent classes do not contribute, so a tile that contains
    only a subset of the classes is scored on the classes it actually has.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Integer label maps of identical shape.
    num_classes : int
        Number of classes.
    ignore_index : int, optional
        Target value to drop before scoring.

    Returns
    -------
    float
        Mean of the present-class IoUs. If no class is present after masking
        (for example an all-``ignore_index`` target) the result is ``1.0``.
    """
    ious = per_class_iou(pred, target, num_classes, ignore_index)
    present = ious[~np.isnan(ious)]
    if present.size == 0:
        return 1.0
    return float(np.mean(present))


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Dense ``num_classes x num_classes`` confusion matrix.

    Row ``i`` / column ``j`` counts pixels whose *target* is class ``i`` and
    whose *prediction* is class ``j``; the diagonal therefore holds the correct
    pixels. Counting is vectorised with ``np.bincount`` on the flattened
    ``target * num_classes + pred`` index.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Integer label maps of class indices in ``range(num_classes)``,
        identical shape.
    num_classes : int
        Number of classes; sets the matrix size.
    ignore_index : int, optional
        Pixels whose ``target`` equals this value are dropped before counting.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(num_classes, num_classes)`` (``target`` along
        rows, ``pred`` along columns).

    Raises
    ------
    ValueError
        If shapes differ or ``num_classes`` is not positive.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    keep = _valid_mask(target, ignore_index)
    p = pred.reshape(-1)
    t = target.reshape(-1)
    if keep is not None:
        keep = keep.reshape(-1)
        p = p[keep]
        t = t[keep]
    p = p.astype(np.int64)
    t = t.astype(np.int64)
    index = t * num_classes + p
    counts = np.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int | None = None,
) -> float:
    """Overall pixel accuracy: fraction of pixels predicted correctly.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Integer label maps of identical shape.
    ignore_index : int, optional
        Pixels whose ``target`` equals this value are excluded from both the
        numerator and the denominator.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``. If every pixel is ignored the result is ``1.0``
        (vacuously correct), matching the empty-mask convention used elsewhere.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
    keep = _valid_mask(target, ignore_index)
    p = pred.reshape(-1)
    t = target.reshape(-1)
    if keep is not None:
        keep = keep.reshape(-1)
        p = p[keep]
        t = t[keep]
    if t.size == 0:
        return 1.0
    return float(np.mean(p == t))


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    """Per-class precision from a confusion matrix.

    With ``target`` along rows and ``pred`` along columns, precision for class
    ``k`` is ``cm[k, k] / cm[:, k].sum()`` (correct predictions over everything
    predicted as ``k``).

    Parameters
    ----------
    cm : numpy.ndarray
        Square confusion matrix from :func:`confusion_matrix`.

    Returns
    -------
    numpy.ndarray
        Float array of length ``num_classes``. A class that is never predicted
        has a zero denominator and is reported as ``nan``.
    """
    cm = np.asarray(cm, dtype=np.float64)
    diag = np.diag(cm)
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(predicted > 0, diag / predicted, np.nan)
    return out


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Per-class recall from a confusion matrix.

    With ``target`` along rows and ``pred`` along columns, recall for class
    ``k`` is ``cm[k, k] / cm[k, :].sum()`` (correct predictions over everything
    whose true class is ``k``).

    Parameters
    ----------
    cm : numpy.ndarray
        Square confusion matrix from :func:`confusion_matrix`.

    Returns
    -------
    numpy.ndarray
        Float array of length ``num_classes``. A class absent from the target
        has a zero denominator and is reported as ``nan``.
    """
    cm = np.asarray(cm, dtype=np.float64)
    diag = np.diag(cm)
    actual = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(actual > 0, diag / actual, np.nan)
    return out


def frequency_weighted_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Frequency-weighted IoU: per-class IoU weighted by class prevalence.

    ``FWIoU = sum_k (freq_k * IoU_k)`` where ``freq_k`` is the fraction of valid
    pixels whose true class is ``k``. Classes absent from the target contribute
    zero weight, so they neither help nor hurt.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Integer label maps of identical shape.
    num_classes : int
        Number of classes.
    ignore_index : int, optional
        Target value to drop before scoring.

    Returns
    -------
    float
        Frequency-weighted IoU in ``[0, 1]``. If no valid pixel remains the
        result is ``1.0``.
    """
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    total = cm.sum()
    if total == 0:
        return 1.0
    actual = cm.sum(axis=1).astype(np.float64)  # true pixels per class
    predicted = cm.sum(axis=0).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)
    union = actual + predicted - diag
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, diag / union, 0.0)
    freq = actual / float(total)
    return float(np.sum(freq * iou))


def cohen_kappa(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    """Cohen's kappa: agreement corrected for chance.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement
    (pixel accuracy) and ``p_e`` is the agreement expected if predictions and
    targets were independent with the same marginals.

    Parameters
    ----------
    pred, target : numpy.ndarray
        Integer label maps of identical shape.
    num_classes : int
        Number of classes.
    ignore_index : int, optional
        Target value to drop before scoring.

    Returns
    -------
    float
        Kappa, typically in ``[-1, 1]``. ``1.0`` is perfect agreement and
        ``~0`` is chance-level. When agreement is already forced (``p_e == 1``,
        e.g. a single class) the result is ``1.0`` if observed agreement is also
        perfect, else ``0.0``.
    """
    cm = confusion_matrix(pred, target, num_classes, ignore_index).astype(np.float64)
    total = cm.sum()
    if total == 0:
        return 1.0
    observed = np.trace(cm) / total
    row_marginal = cm.sum(axis=1) / total
    col_marginal = cm.sum(axis=0) / total
    expected = float(np.sum(row_marginal * col_marginal))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))
