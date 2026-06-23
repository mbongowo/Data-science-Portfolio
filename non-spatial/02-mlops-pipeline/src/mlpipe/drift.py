"""Data-drift detection in pure numpy / pandas — the differentiated core.

A trained model is only as good as the assumption that tomorrow's inputs look
like the data it was fit on. When the input distribution shifts (a wetter
season, a recalibrated sensor, a new station), accuracy quietly rots while every
service-health dashboard stays green. This module is the cheap, dependency-free
early-warning system that most portfolios leave out.

Two classic two-sample tests live here, both computed from samples alone with no
SciPy:

* :func:`psi` — the **Population Stability Index**, the industry-standard scalar
  for "how much has this feature's distribution moved". Reference data is binned
  at its own quantiles; the index sums ``(c - r) * ln(c / r)`` over bins, where
  ``r`` and ``c`` are the reference / current bin proportions. It is symmetric in
  a useful sense and grows smoothly with the size of the shift. The conventional
  reading is:

  =================  ==========================================
  PSI                interpretation
  =================  ==========================================
  ``< 0.1``          no significant population change
  ``0.1 - 0.2``      moderate shift — worth a look
  ``>= 0.2``         major shift — investigate / consider retrain
  =================  ==========================================

* :func:`ks_statistic` — the two-sample **Kolmogorov-Smirnov** ``D`` statistic,
  the largest vertical gap between the two empirical CDFs. ``D = 0`` means the
  samples are identical; ``D = 1`` means they do not overlap at all.

:func:`feature_drift_report` runs both per feature across two frames and returns
a tidy, sortable table plus a drifted-feature summary — the runnable fallback
that :mod:`mlpipe.monitor` (Evidently) dresses up for a dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike

#: Conventional PSI thresholds (see the module docstring).
PSI_MINOR = 0.1
PSI_MAJOR = 0.2


def psi(reference: ArrayLike, current: ArrayLike, bins: int = 10) -> float:
    r"""Population Stability Index between two 1-D samples.

    The reference sample defines ``bins`` quantile bins (so each reference bin
    holds roughly the same count). Both samples are histogrammed into those
    fixed edges and converted to proportions; the index is

    .. math::

        \mathrm{PSI} = \sum_{i} (c_i - r_i)\,\ln\!\frac{c_i}{r_i},

    where ``r_i`` / ``c_i`` are the reference / current proportions in bin ``i``.
    Empty bins are floored at a small ``epsilon`` so the log and ratio stay
    finite. PSI is ``~0`` for two samples from the same distribution and grows
    with the shift; see the module docstring for the 0.1 / 0.2 thresholds.

    Parameters
    ----------
    reference, current:
        1-D numeric samples. They need not be the same length.
    bins:
        Number of quantile bins taken from the reference sample (default 10).

    Returns
    -------
    float
        The PSI (always ``>= 0``).

    Raises
    ------
    ValueError
        If either sample is empty or ``bins < 1``.
    """
    r = np.asarray(reference, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()
    if r.size == 0 or c.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples.")
    if bins < 1:
        raise ValueError("bins must be a positive integer.")

    # Quantile edges from the reference; collapse duplicate edges (constant
    # regions) so we never create a zero-width bin.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(r, quantiles))
    if edges.size < 2:
        # Reference is (numerically) constant: only divergence is current spread.
        return 0.0 if np.allclose(c, edges[0]) else float("inf")

    # Open the outer edges so points outside the reference range still land in
    # the end bins rather than being dropped.
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    r_counts, _ = np.histogram(r, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)

    eps = 1e-6
    r_prop = np.clip(r_counts / r.size, eps, None)
    c_prop = np.clip(c_counts / c.size, eps, None)

    return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))


def ks_statistic(reference: ArrayLike, current: ArrayLike) -> float:
    r"""Two-sample Kolmogorov-Smirnov ``D`` statistic (max ECDF gap).

    ``D`` is the supremum over the real line of ``|F_ref(x) - F_cur(x)|``, the
    two empirical cumulative distribution functions. It is computed exactly from
    the samples: evaluate both ECDFs at every observed point in the pooled set
    and take the largest absolute difference.

    Parameters
    ----------
    reference, current:
        1-D numeric samples (any lengths).

    Returns
    -------
    float
        ``D`` in ``[0, 1]``. ``0`` for identical samples; ``1`` for samples with
        disjoint support.

    Raises
    ------
    ValueError
        If either sample is empty.
    """
    r = np.sort(np.asarray(reference, dtype=float).ravel())
    c = np.sort(np.asarray(current, dtype=float).ravel())
    if r.size == 0 or c.size == 0:
        raise ValueError("KS needs non-empty reference and current samples.")

    pooled = np.concatenate([r, c])
    # ECDF value just to the right of each pooled point: fraction of sample <= x.
    cdf_r = np.searchsorted(r, pooled, side="right") / r.size
    cdf_c = np.searchsorted(c, pooled, side="right") / c.size
    return float(np.max(np.abs(cdf_r - cdf_c)))


def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = PSI_MAJOR,
) -> pd.DataFrame:
    """Per-feature PSI / KS drift report across two frames.

    For every shared column the report computes :func:`psi` and
    :func:`ks_statistic` between the reference and current frame and flags the
    feature as drifted when its PSI reaches ``psi_threshold`` (default 0.2, the
    "major shift" line). The returned frame carries a ``summary`` attribute via
    :attr:`pandas.DataFrame.attrs` with the drifted count.

    Parameters
    ----------
    reference_df, current_df:
        Frames with the **same columns** (the features to monitor). Only numeric
        columns are scored.
    psi_threshold:
        PSI value at or above which a feature is marked ``drifted``.

    Returns
    -------
    pandas.DataFrame
        One row per feature with columns ``feature``, ``psi``, ``ks``, and
        ``drifted`` (bool), sorted by descending PSI. ``.attrs["summary"]`` holds
        ``{"n_features", "n_drifted", "drifted_features"}``.

    Raises
    ------
    ValueError
        If the frames are empty or do not share the same set of columns.
    """
    if reference_df.empty or current_df.empty:
        raise ValueError("Drift report needs non-empty reference and current frames.")
    if set(reference_df.columns) != set(current_df.columns):
        raise ValueError(
            "reference_df and current_df must have the same columns; got "
            f"{sorted(reference_df.columns)} vs {sorted(current_df.columns)}."
        )

    rows: list[dict[str, object]] = []
    for col in reference_df.columns:
        ref_col = pd.to_numeric(reference_df[col], errors="coerce").dropna().to_numpy()
        cur_col = pd.to_numeric(current_df[col], errors="coerce").dropna().to_numpy()
        if ref_col.size == 0 or cur_col.size == 0:
            continue
        col_psi = psi(ref_col, cur_col)
        col_ks = ks_statistic(ref_col, cur_col)
        rows.append(
            {
                "feature": col,
                "psi": round(col_psi, 6),
                "ks": round(col_ks, 6),
                "drifted": bool(col_psi >= psi_threshold),
            }
        )

    report = pd.DataFrame(rows, columns=["feature", "psi", "ks", "drifted"])
    report = report.sort_values("psi", ascending=False).reset_index(drop=True)

    drifted = report.loc[report["drifted"], "feature"].tolist()
    report.attrs["summary"] = {
        "n_features": int(len(report)),
        "n_drifted": int(len(drifted)),
        "drifted_features": drifted,
    }
    return report
