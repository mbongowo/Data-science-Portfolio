"""Alternating Least Squares matrix factorisation (pure numpy reference).

This module factorises a ratings matrix ``R`` (shape ``m`` users x ``n`` items)
into a user-factor matrix ``U`` (``m`` x ``rank``) and an item-factor matrix
``V`` (``n`` x ``rank``) such that ``U @ V.T`` approximates ``R`` on the observed
entries. Only entries flagged by a boolean ``mask`` are fit; the rest are
treated as missing, which is the whole point of collaborative filtering.

It is a dependency-free reference (numpy only) used for validation, teaching,
and small problems. It solves the explicit-feedback ALS objective

.. math::

    \\min_{U, V} \\sum_{(i, j) \\in \\Omega} (R_{ij} - u_i^\\top v_j)^2
        + \\lambda \\left( \\sum_i \\lVert u_i \\rVert^2
                          + \\sum_j \\lVert v_j \\rVert^2 \\right)

where :math:`\\Omega` is the observed set (the ``mask``) and :math:`\\lambda` is
the ridge penalty ``reg``. Holding ``V`` fixed, each user row is a ridge
regression on that user's observed items; holding ``U`` fixed, each item row is
a ridge regression on that item's observed users. Alternating these two closed-
form solves is guaranteed not to increase the objective, so it converges to a
local optimum.

For a distributed implementation on data that does not fit in memory, see
:mod:`recsys.spark_als`, which wraps Spark MLlib's ALS. The maths is the same;
only the execution engine differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray


def _solve_factors(
    fixed: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[np.bool_],
    reg: float,
) -> NDArray[np.float64]:
    """Ridge-solve one side of the factorisation.

    For each row ``i`` of the side being updated, fit a ridge regression of the
    observed targets ``target[i, observed]`` on the fixed factors of those
    observed columns. Returns the updated factor matrix (rows x rank).
    """
    n_rows = target.shape[0]
    rank = fixed.shape[1]
    out = np.zeros((n_rows, rank), dtype=float)
    eye = reg * np.eye(rank)
    for i in range(n_rows):
        observed = mask[i]
        if not observed.any():
            continue  # no data for this row -> leave factors at zero
        f = fixed[observed]  # (n_obs, rank)
        r = target[i, observed]  # (n_obs,)
        a = f.T @ f + eye  # (rank, rank), regularised
        b = f.T @ r  # (rank,)
        out[i] = np.linalg.solve(a, b)
    return out


def als_factorize(
    R: ArrayLike,
    mask: ArrayLike,
    rank: int,
    reg: float = 0.1,
    iters: int = 15,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Factorise ``R`` over the observed ``mask`` via alternating ridge solves.

    Parameters
    ----------
    R:
        Ratings matrix, shape ``(m, n)``. Values where ``mask`` is False are
        ignored (they may be anything, including zero or NaN-free placeholders).
    mask:
        Boolean array the same shape as ``R``. True marks an observed entry.
    rank:
        Latent dimension (number of factors). Must be >= 1.
    reg:
        L2 ridge penalty ``lambda`` on the factors. Larger => more shrinkage.
    iters:
        Number of alternating sweeps (one sweep updates U then V).
    seed:
        Seed for the random initialisation of the item factors. Fixing it makes
        the result reproducible.

    Returns
    -------
    (U, V):
        ``U`` has shape ``(m, rank)`` and ``V`` has shape ``(n, rank)``. The
        reconstruction is ``U @ V.T`` (see :func:`predict`).

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``rank < 1`` or ``iters < 1``.

    Notes
    -----
    On a fully observed low-rank matrix the factorisation reconstructs ``R`` to
    a very small RMSE (this is the known-answer test). With ``reg = 0`` it is
    exact up to numerical error; a small ``reg`` introduces a little shrinkage.
    """
    ratings = np.asarray(R, dtype=float)
    observed = np.asarray(mask, dtype=bool)

    if ratings.ndim != 2:
        raise ValueError("R must be a 2-D matrix.")
    if observed.shape != ratings.shape:
        raise ValueError(
            f"mask shape {observed.shape} does not match R shape {ratings.shape}."
        )
    if rank < 1:
        raise ValueError("rank must be >= 1.")
    if iters < 1:
        raise ValueError("iters must be >= 1.")

    m, n = ratings.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank)) * 0.1
    V = rng.standard_normal((n, rank)) * 0.1

    for _ in range(iters):
        U = _solve_factors(V, ratings, observed, reg)
        V = _solve_factors(U, ratings.T, observed.T, reg)

    return U, V


def predict(U: ArrayLike, V: ArrayLike) -> NDArray[np.float64]:
    """Reconstruct the dense score matrix ``U @ V.T``.

    Parameters
    ----------
    U:
        User-factor matrix, shape ``(m, rank)``.
    V:
        Item-factor matrix, shape ``(n, rank)``.

    Returns
    -------
    numpy.ndarray
        The ``(m, n)`` matrix of predicted scores. Higher means a stronger
        predicted preference; rank a user's row to get their top-N items.
    """
    u = np.asarray(U, dtype=float)
    v = np.asarray(V, dtype=float)
    return u @ v.T


def als_factorize_biased(
    R: ArrayLike,
    mask: ArrayLike,
    rank: int,
    reg: float = 0.1,
    iters: int = 15,
    seed: int | None = 0,
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""Biased ALS: latent factors **plus** global / user / item bias terms.

    Many real ratings matrices are dominated by *additive* effects rather than
    interaction structure: some users rate everything high, some items are
    universally loved. A plain factorisation has to spend latent dimensions
    explaining those offsets. The biased model factors them out explicitly,

    .. math::

        \hat{R}_{ij} = \mu + b^u_i + b^v_j + u_i^\top v_j

    and fits the regularised squared error over the observed set :math:`\Omega`

    .. math::

        \min \sum_{(i,j)\in\Omega} (R_{ij} - \hat{R}_{ij})^2
            + \lambda\big(\lVert u_i\rVert^2 + \lVert v_j\rVert^2
                          + (b^u_i)^2 + (b^v_j)^2\big).

    The global mean :math:`\mu` is the mean of the observed ratings (fixed once).
    Each alternating sweep then updates, in closed form per row:

    * user biases :math:`b^u_i` (ridge mean of that user's observed residuals),
    * item biases :math:`b^v_j`,
    * user factors :math:`u_i` (ridge regression of the residual
      :math:`R_{ij}-\mu-b^u_i-b^v_j` on the observed item factors),
    * item factors :math:`v_j`.

    Parameters
    ----------
    R, mask:
        Ratings matrix and boolean observed mask, both shape ``(m, n)``.
    rank:
        Latent dimension (>= 1).
    reg:
        Shared L2 penalty applied to factors **and** bias terms.
    iters:
        Number of alternating sweeps.
    seed:
        Seed for the factor initialisation.

    Returns
    -------
    (mu, bu, bv, U, V):
        Global mean scalar, user-bias vector ``(m,)``, item-bias vector
        ``(n,)``, user factors ``(m, rank)`` and item factors ``(n, rank)``.
        Reconstruct with :func:`predict_biased`.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``rank < 1``, ``iters < 1``, or the mask has
        no observed entries (the global mean would be undefined).

    Notes
    -----
    On data whose signal is mostly additive user/item offsets, this model
    reaches a lower RMSE than the unbiased :func:`als_factorize` at the same
    rank, because the biases absorb the offsets directly instead of forcing the
    factors to approximate them. That improvement is the known-answer test.
    """
    ratings = np.asarray(R, dtype=float)
    observed = np.asarray(mask, dtype=bool)

    if ratings.ndim != 2:
        raise ValueError("R must be a 2-D matrix.")
    if observed.shape != ratings.shape:
        raise ValueError(
            f"mask shape {observed.shape} does not match R shape {ratings.shape}."
        )
    if rank < 1:
        raise ValueError("rank must be >= 1.")
    if iters < 1:
        raise ValueError("iters must be >= 1.")
    if not observed.any():
        raise ValueError("mask has no observed entries; mu is undefined.")

    m, n = ratings.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank)) * 0.1
    V = rng.standard_normal((n, rank)) * 0.1

    mu = float(ratings[observed].mean())
    bu = np.zeros(m, dtype=float)
    bv = np.zeros(n, dtype=float)

    for _ in range(iters):
        # --- user biases: ridge mean of residual R - mu - bv - u.v over the
        # user's observed items. Closed form: sum(resid) / (n_obs + reg).
        for i in range(m):
            obs = observed[i]
            n_obs = int(obs.sum())
            if n_obs == 0:
                bu[i] = 0.0
                continue
            resid = ratings[i, obs] - mu - bv[obs] - V[obs] @ U[i]
            bu[i] = resid.sum() / (n_obs + reg)

        # --- item biases (symmetric) ---
        for j in range(n):
            obs = observed[:, j]
            n_obs = int(obs.sum())
            if n_obs == 0:
                bv[j] = 0.0
                continue
            resid = ratings[obs, j] - mu - bu[obs] - U[obs] @ V[j]
            bv[j] = resid.sum() / (n_obs + reg)

        # --- user factors: ridge-regress the bias-removed residual on items ---
        eye = reg * np.eye(rank)
        for i in range(m):
            obs = observed[i]
            if not obs.any():
                U[i] = 0.0
                continue
            f = V[obs]
            target = ratings[i, obs] - mu - bu[i] - bv[obs]
            U[i] = np.linalg.solve(f.T @ f + eye, f.T @ target)

        # --- item factors (symmetric) ---
        for j in range(n):
            obs = observed[:, j]
            if not obs.any():
                V[j] = 0.0
                continue
            f = U[obs]
            target = ratings[obs, j] - mu - bu[obs] - bv[j]
            V[j] = np.linalg.solve(f.T @ f + eye, f.T @ target)

    return mu, bu, bv, U, V


def predict_biased(
    mu: float,
    bu: ArrayLike,
    bv: ArrayLike,
    U: ArrayLike,
    V: ArrayLike,
) -> NDArray[np.float64]:
    r"""Reconstruct the biased model's dense score matrix.

    Returns :math:`\hat{R}_{ij} = \mu + b^u_i + b^v_j + u_i^\top v_j` as an
    ``(m, n)`` array, i.e. the global mean plus the per-user and per-item bias
    broadcast over the factor interaction term.

    Parameters
    ----------
    mu:
        Global mean scalar.
    bu, bv:
        User-bias ``(m,)`` and item-bias ``(n,)`` vectors.
    U, V:
        User ``(m, rank)`` and item ``(n, rank)`` factor matrices.

    Returns
    -------
    numpy.ndarray
        The ``(m, n)`` predicted-score matrix.
    """
    u = np.asarray(U, dtype=float)
    v = np.asarray(V, dtype=float)
    bu_a = np.asarray(bu, dtype=float)
    bv_a = np.asarray(bv, dtype=float)
    return float(mu) + bu_a[:, None] + bv_a[None, :] + u @ v.T


def als_implicit(
    C: ArrayLike,
    P: ArrayLike,
    rank: int,
    reg: float = 0.1,
    iters: int = 15,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Implicit-feedback ALS (Hu-Koren-Volinsky confidence-weighted form).

    Implicit feedback has no ratings, only *that* an interaction happened (a
    play, a click). The HKV formulation turns counts into a binary preference
    ``P`` (1 if any interaction, else 0) and a per-cell **confidence** ``C``
    (how strongly we trust that preference; larger for more interactions). Every
    user-item cell is fit — the zeros are real "no preference, low confidence"
    signal, not missing data:

    .. math::

        \min_{U,V} \sum_{i,j} C_{ij}\,(P_{ij} - u_i^\top v_j)^2
            + \lambda\big(\sum_i\lVert u_i\rVert^2 + \sum_j\lVert v_j\rVert^2\big).

    Holding ``V`` fixed, each user solve has the closed form

    .. math::

        u_i = \big(V^\top C^i V + \lambda I\big)^{-1} V^\top C^i p_i,

    where :math:`C^i = \mathrm{diag}(C_{i,\cdot})`. We form ``V.T @ (c[:,None]*V)``
    directly (no dense diagonal), which is the standard small-scale version of
    the HKV trick.

    Parameters
    ----------
    C:
        Confidence matrix, shape ``(m, n)``, non-negative. A common choice is
        ``C = 1 + alpha * counts``.
    P:
        Binary preference matrix, shape ``(m, n)``, entries in ``{0, 1}``.
    rank:
        Latent dimension (>= 1).
    reg:
        L2 penalty on the factors.
    iters:
        Number of alternating sweeps.
    seed:
        Seed for the factor initialisation.

    Returns
    -------
    (U, V):
        User factors ``(m, rank)`` and item factors ``(n, rank)``. The predicted
        preference is ``U @ V.T`` (use :func:`predict`); rank a user's row for a
        top-N recommendation.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``rank < 1``, or ``iters < 1``.

    Notes
    -----
    On a clean two-block binary preference matrix (two user groups, each liking
    a disjoint block of items) the learned scores are systematically higher on
    the preferred block than off it — that separation is the known-answer test.
    """
    conf = np.asarray(C, dtype=float)
    pref = np.asarray(P, dtype=float)

    if conf.ndim != 2:
        raise ValueError("C must be a 2-D matrix.")
    if pref.shape != conf.shape:
        raise ValueError(
            f"P shape {pref.shape} does not match C shape {conf.shape}."
        )
    if rank < 1:
        raise ValueError("rank must be >= 1.")
    if iters < 1:
        raise ValueError("iters must be >= 1.")

    m, n = conf.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank)) * 0.1
    V = rng.standard_normal((n, rank)) * 0.1
    eye = reg * np.eye(rank)

    def _update(
        factors: NDArray[np.float64],
        c_rows: NDArray[np.float64],
        p_rows: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Solve for each row given the fixed `factors` (n_cols x rank).
        out = np.zeros((c_rows.shape[0], rank), dtype=float)
        for i in range(c_rows.shape[0]):
            c = c_rows[i]  # (n_cols,)
            # weighted Gram: factors.T @ diag(c) @ factors
            a = factors.T @ (c[:, None] * factors) + eye
            b = factors.T @ (c * p_rows[i])
            out[i] = np.linalg.solve(a, b)
        return out

    for _ in range(iters):
        U = _update(V, conf, pref)
        V = _update(U, conf.T, pref.T)

    return U, V
