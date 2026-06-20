"""Exploratory spatial data analysis: global and local autocorrelation.

This module exposes two kinds of functions:

1. A **pure-numpy reference layer** with no third-party dependency, so it is
   always importable and is the basis of the known-answer unit tests. It covers
   global Moran's I (:func:`morans_i_dense`), Geary's C (:func:`gearys_c_dense`),
   Local Moran's I / LISA (:func:`local_moran_dense`, :func:`lisa_quadrants`),
   and standardised Getis-Ord Gi* (:func:`getis_ord_g_star_dense`). These return
   point estimates only; use them for small problems, teaching, and validation.

2. **pysal/esda wrappers** (:func:`global_moran`, :func:`local_moran`,
   :func:`getis_ord_gi_star`) that call the ``esda`` library for
   permutation-based pseudo p-values and the conditional-randomisation
   inference that real analyses require. These import ``esda`` lazily so that
   the rest of the package (and the test suite) works without it installed.

Interpretation notes (do not skip these):

* Moran's I measures *global* spatial autocorrelation under the assumption that
  the spatial process is stationary across the study area. A single global
  statistic can hide strong local structure.
* LISA (Local Moran's I) and Gi* are computed at every location and are highly
  multiple-tested. The pseudo p-values are *conditional* and should be treated
  as descriptive flags, optionally corrected (e.g. FDR / Bonferroni). They do
  not establish a causal mechanism.
* Inference is permutation-based and therefore depends on the chosen number of
  permutations and the random seed. Report both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


# --------------------------------------------------------------------------- #
# Pure-numpy reference implementation (no third-party deps)
# --------------------------------------------------------------------------- #
def morans_i_dense(values: "ArrayLike", w: "ArrayLike") -> float:
    r"""Compute global Moran's I from a dense weights matrix.

    Implements the textbook estimator

    .. math::

        I = \frac{n}{S_0} \cdot \frac{z^\top W z}{z^\top z}

    where :math:`z = x - \bar{x}` are the mean-centred values, :math:`W` is the
    (``n`` x ``n``) spatial weights matrix, :math:`S_0 = \sum_{i,j} w_{ij}` is
    the sum of all weights, and ``n`` is the number of observations.

    This is a deterministic, dependency-free reference used for validation and
    teaching. For real inference (pseudo p-values), use :func:`global_moran`.

    Parameters
    ----------
    values:
        Length-``n`` array of observed values.
    w:
        Dense ``n`` x ``n`` spatial weights matrix. The diagonal is ignored
        (self-neighbours are not meaningful); it need not be row-standardised.

    Returns
    -------
    float
        Moran's I. Positive => like values cluster; near :math:`-1/(n-1)` =>
        no autocorrelation; negative => checkerboard / dispersion.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``n < 2``, the weights sum to zero, or the
        values have zero variance (I is undefined).
    """
    x = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(w, dtype=float)

    n = x.size
    if n < 2:
        raise ValueError("Moran's I requires at least 2 observations.")
    if weights.shape != (n, n):
        raise ValueError(
            f"Weights matrix shape {weights.shape} does not match "
            f"the number of observations ({n})."
        )

    weights = weights.copy()
    np.fill_diagonal(weights, 0.0)

    s0 = float(weights.sum())
    if s0 == 0.0:
        raise ValueError("Sum of weights (S0) is zero; every unit is an island.")

    z = x - x.mean()
    denom = float(z @ z)
    if denom == 0.0:
        raise ValueError("Values have zero variance; Moran's I is undefined.")

    numer = float(z @ weights @ z)
    return (n / s0) * (numer / denom)


def expected_morans_i(n: int) -> float:
    """Return the expected value of Moran's I under the null, ``-1/(n-1)``."""
    if n < 2:
        raise ValueError("n must be >= 2.")
    return -1.0 / (n - 1)


def gearys_c_dense(values: "ArrayLike", w: "ArrayLike") -> float:
    r"""Compute Geary's C from a dense weights matrix.

    Implements the textbook estimator

    .. math::

        C = \frac{n - 1}{2 S_0} \cdot
            \frac{\sum_{i,j} w_{ij} (x_i - x_j)^2}{\sum_i (x_i - \bar{x})^2}

    where :math:`S_0 = \sum_{i,j} w_{ij}` and ``n`` is the number of
    observations. Geary's C is a distance-based companion to Moran's I. Its
    null expectation is 1. Values below 1 indicate positive autocorrelation
    (neighbouring values are similar); values above 1 indicate negative
    autocorrelation (neighbouring values differ). C is more sensitive to local
    differences than Moran's I, which is built on cross-products of deviations
    from the mean.

    This is a deterministic, dependency-free reference for validation and
    teaching; it returns the point estimate only, with no inference.

    Parameters
    ----------
    values:
        Length-``n`` array of observed values.
    w:
        Dense ``n`` x ``n`` spatial weights matrix. The diagonal is ignored;
        it need not be row-standardised.

    Returns
    -------
    float
        Geary's C. Near 1 => no autocorrelation; below 1 => clustering;
        above 1 => dispersion.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``n < 2``, the weights sum to zero, or the
        values have zero variance (C is undefined).
    """
    x = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(w, dtype=float)

    n = x.size
    if n < 2:
        raise ValueError("Geary's C requires at least 2 observations.")
    if weights.shape != (n, n):
        raise ValueError(
            f"Weights matrix shape {weights.shape} does not match "
            f"the number of observations ({n})."
        )

    weights = weights.copy()
    np.fill_diagonal(weights, 0.0)

    s0 = float(weights.sum())
    if s0 == 0.0:
        raise ValueError("Sum of weights (S0) is zero; every unit is an island.")

    z = x - x.mean()
    denom = float(z @ z)
    if denom == 0.0:
        raise ValueError("Values have zero variance; Geary's C is undefined.")

    # sum_ij w_ij (x_i - x_j)^2 = sum_ij w_ij (x_i^2 - 2 x_i x_j + x_j^2)
    diff_sq = (x[:, None] - x[None, :]) ** 2
    numer = float((weights * diff_sq).sum())
    return (n - 1) / (2.0 * s0) * (numer / denom)


def _row_standardize(weights: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Return ``weights`` with each non-empty row scaled to sum to one."""
    row_sums = weights.sum(axis=1, keepdims=True)
    out = np.zeros_like(weights)
    nonzero = row_sums.ravel() != 0.0
    out[nonzero] = weights[nonzero] / row_sums[nonzero]
    return out


def local_moran_dense(
    values: "ArrayLike", w: "ArrayLike", *, row_standardize: bool = True
) -> "NDArray[np.float64]":
    r"""Compute Local Moran's I (LISA) from a dense weights matrix.

    For each location ``i``,

    .. math::

        I_i = \frac{z_i}{m_2} \sum_j w_{ij} z_j,
        \qquad z = x - \bar{x},
        \qquad m_2 = \frac{1}{n} \sum_i z_i^2 .

    With a row-standardised ``W`` the inner sum is the spatial lag (the local
    average of the centred neighbours). The local statistics sum to ``n`` times
    the global Moran's I, and with a row-standardised ``W`` their mean equals
    the global I. This is the property the test suite checks.

    Parameters
    ----------
    values:
        Length-``n`` array of observed values.
    w:
        Dense ``n`` x ``n`` spatial weights matrix. The diagonal is ignored.
    row_standardize:
        If True (default), scale each row of ``W`` to sum to one before forming
        the spatial lag, which is the usual convention for LISA.

    Returns
    -------
    numpy.ndarray
        Length-``n`` array of local Moran statistics. The sign of ``I_i``
        combined with the sign of the spatial lag gives the LISA quadrant; see
        :func:`lisa_quadrants`.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``n < 2``, or the values have zero variance.
    """
    x = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(w, dtype=float)

    n = x.size
    if n < 2:
        raise ValueError("Local Moran's I requires at least 2 observations.")
    if weights.shape != (n, n):
        raise ValueError(
            f"Weights matrix shape {weights.shape} does not match "
            f"the number of observations ({n})."
        )

    weights = weights.copy()
    np.fill_diagonal(weights, 0.0)
    if row_standardize:
        weights = _row_standardize(weights)

    z = x - x.mean()
    m2 = float(z @ z) / n
    if m2 == 0.0:
        raise ValueError("Values have zero variance; Local Moran's I is undefined.")

    lag = weights @ z
    return (z / m2) * lag


def lisa_quadrants(
    values: "ArrayLike", w: "ArrayLike", *, row_standardize: bool = True
) -> "NDArray[np.str_]":
    """Assign each location to its LISA quadrant.

    The quadrant compares a location's own centred value ``z_i`` against its
    spatial lag (the centred-value average of its neighbours):

    * ``HH`` high value, high-valued neighbours (hot cluster)
    * ``LL`` low value, low-valued neighbours (cold cluster)
    * ``LH`` low value, high-valued neighbours (spatial outlier)
    * ``HL`` high value, low-valued neighbours (spatial outlier)

    A location whose own value or whose spatial lag is exactly at the mean is
    labelled ``00`` (undefined quadrant). This is the deterministic label only;
    it carries no significance test. For inference use :func:`local_moran`.

    Parameters
    ----------
    values:
        Length-``n`` array of observed values.
    w:
        Dense ``n`` x ``n`` spatial weights matrix. The diagonal is ignored.
    row_standardize:
        If True (default), row-standardise ``W`` before forming the lag.

    Returns
    -------
    numpy.ndarray
        Length-``n`` array of quadrant labels from
        ``{"HH", "LL", "LH", "HL", "00"}``.
    """
    x = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(w, dtype=float)

    n = x.size
    if weights.shape != (n, n):
        raise ValueError(
            f"Weights matrix shape {weights.shape} does not match "
            f"the number of observations ({n})."
        )

    weights = weights.copy()
    np.fill_diagonal(weights, 0.0)
    if row_standardize:
        weights = _row_standardize(weights)

    z = x - x.mean()
    lag = weights @ z

    labels = np.full(n, "00", dtype="<U2")
    labels[(z > 0) & (lag > 0)] = "HH"
    labels[(z < 0) & (lag < 0)] = "LL"
    labels[(z < 0) & (lag > 0)] = "LH"
    labels[(z > 0) & (lag < 0)] = "HL"
    return labels


def getis_ord_g_star_dense(values: "ArrayLike", w: "ArrayLike") -> "NDArray[np.float64]":
    r"""Compute standardised Getis-Ord Gi* z-scores from a dense weights matrix.

    The focal unit is included in its own neighbourhood (the ``*`` in Gi*), so
    the diagonal of ``W`` is set to one before the statistic is formed. For each
    location ``i`` the standardised score is

    .. math::

        G_i^* = \frac{\sum_j w_{ij} x_j - \bar{X} \sum_j w_{ij}}
                     {S \sqrt{\dfrac{n \sum_j w_{ij}^2 - (\sum_j w_{ij})^2}{n - 1}}}

    where :math:`\bar{X}` is the global mean and
    :math:`S = \sqrt{\frac{1}{n}\sum_j x_j^2 - \bar{X}^2}` is the population
    standard deviation. Large positive scores mark hot spots (a unit and its
    neighbours hold high values); large negative scores mark cold spots.

    The weights passed in are used as supplied (after forcing a unit diagonal);
    pass a binary contiguity matrix for the conventional Gi*. This returns the
    asymptotic z-scores only; for a pseudo p-value use :func:`getis_ord_gi_star`.

    Parameters
    ----------
    values:
        Length-``n`` array of observed values.
    w:
        Dense ``n`` x ``n`` spatial weights matrix. The diagonal is overwritten
        with ones so each unit is its own neighbour.

    Returns
    -------
    numpy.ndarray
        Length-``n`` array of Gi* z-scores.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``n < 3`` (the variance term needs
        ``n - 1`` and a non-degenerate window), or the values have zero
        variance.
    """
    x = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(w, dtype=float)

    n = x.size
    if n < 3:
        raise ValueError("Gi* requires at least 3 observations.")
    if weights.shape != (n, n):
        raise ValueError(
            f"Weights matrix shape {weights.shape} does not match "
            f"the number of observations ({n})."
        )

    weights = weights.copy()
    np.fill_diagonal(weights, 1.0)  # star: include the focal unit

    x_bar = float(x.mean())
    s = float(np.sqrt((x @ x) / n - x_bar**2))
    if s == 0.0:
        raise ValueError("Values have zero variance; Gi* is undefined.")

    w_sum = weights.sum(axis=1)
    w_sq = (weights**2).sum(axis=1)
    numer = weights @ x - x_bar * w_sum
    var = (n * w_sq - w_sum**2) / (n - 1)
    return numer / (s * np.sqrt(var))


# --------------------------------------------------------------------------- #
# Result containers for the pysal-backed analyses
# --------------------------------------------------------------------------- #
@dataclass
class GlobalMoranResult:
    """Result of a global Moran's I test."""

    I: float
    expected_I: float
    p_sim: float
    z_sim: float
    permutations: int


@dataclass
class LocalResult:
    """Result of a local (per-observation) ESDA statistic."""

    statistic: "NDArray[np.float64]"
    p_sim: "NDArray[np.float64]"
    labels: "NDArray[np.str_]"
    significant: "NDArray[np.bool_]"


# --------------------------------------------------------------------------- #
# pysal / esda wrappers (lazy import)
# --------------------------------------------------------------------------- #
def global_moran(
    values: "ArrayLike",
    w: Any,
    *,
    permutations: int = 999,
    seed: int | None = 42,
) -> GlobalMoranResult:
    """Global Moran's I with permutation inference via ``esda``.

    Parameters
    ----------
    values:
        Length-``n`` array of the variable of interest.
    w:
        A ``libpysal`` ``W`` object (ideally row-standardised).
    permutations:
        Number of conditional permutations for the pseudo p-value.
    seed:
        Seed for reproducible permutations.

    Returns
    -------
    GlobalMoranResult
    """
    from esda.moran import Moran  # lazy import

    if seed is not None:
        np.random.seed(seed)

    mi = Moran(np.asarray(values, dtype=float), w, permutations=permutations)
    return GlobalMoranResult(
        I=float(mi.I),
        expected_I=float(mi.EI),
        p_sim=float(mi.p_sim),
        z_sim=float(mi.z_sim),
        permutations=permutations,
    )


def moran_scatterplot(values: "ArrayLike", w: Any, ax: Any = None) -> Any:
    """Draw a Moran scatterplot using ``splot``; returns the matplotlib axis."""
    from esda.moran import Moran
    from splot.esda import moran_scatterplot as _scatter

    mi = Moran(np.asarray(values, dtype=float), w)
    _, ax = _scatter(mi, ax=ax)
    return ax


# LISA quadrant codes used by esda.Moran_Local
_LISA_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def local_moran(
    values: "ArrayLike",
    w: Any,
    *,
    permutations: int = 999,
    significance: float = 0.05,
    seed: int | None = 42,
) -> LocalResult:
    """Local Moran's I (LISA) with significance masking via ``esda``.

    Labels follow the standard LISA quadrants:

    * ``HH`` high value surrounded by high (hot cluster)
    * ``LL`` low surrounded by low (cold cluster)
    * ``LH`` low surrounded by high (spatial outlier)
    * ``HL`` high surrounded by low (spatial outlier)
    * ``ns`` not significant at the chosen level

    Only locations with ``p_sim <= significance`` keep their quadrant label;
    everything else is labelled ``ns``.
    """
    from esda.moran import Moran_Local

    if seed is not None:
        np.random.seed(seed)

    lm = Moran_Local(
        np.asarray(values, dtype=float), w, permutations=permutations
    )
    p = np.asarray(lm.p_sim, dtype=float)
    sig = p <= significance
    labels = np.array(
        [
            _LISA_LABELS.get(int(q), "ns") if s else "ns"
            for q, s in zip(lm.q, sig)
        ],
        dtype=object,
    ).astype(str)

    return LocalResult(
        statistic=np.asarray(lm.Is, dtype=float),
        p_sim=p,
        labels=labels,
        significant=sig,
    )


def getis_ord_gi_star(
    values: "ArrayLike",
    w: Any,
    *,
    permutations: int = 999,
    significance: float = 0.05,
    seed: int | None = 42,
) -> LocalResult:
    """Getis-Ord Gi* hot/cold spot statistic via ``esda``.

    ``star=True`` includes the focal unit in its own neighbourhood, which is
    the conventional Gi* (as opposed to Gi). Significant positive z-scores are
    labelled ``hot`` and significant negative z-scores ``cold``; the rest
    ``ns``.
    """
    from esda.getisord import G_Local

    if seed is not None:
        np.random.seed(seed)

    gi = G_Local(
        np.asarray(values, dtype=float),
        w,
        star=True,
        permutations=permutations,
    )
    z = np.asarray(gi.Zs, dtype=float)
    p = np.asarray(gi.p_sim, dtype=float)
    sig = p <= significance
    labels = np.where(sig & (z > 0), "hot", np.where(sig & (z < 0), "cold", "ns"))

    return LocalResult(
        statistic=z,
        p_sim=p,
        labels=labels.astype(str),
        significant=sig,
    )
