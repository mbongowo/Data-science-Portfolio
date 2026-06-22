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
