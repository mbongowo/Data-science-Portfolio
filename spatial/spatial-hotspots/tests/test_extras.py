"""Known-answer tests for the extra pure-numpy ESDA capabilities.

Covers join counts, bivariate Moran's I, the Moran-scatter-slope identity, the
Benjamini-Hochberg FDR correction, and the grid weight helpers. Expected values
are hand-derived on tiny graphs and documented inline, so a green test proves
the estimator is correct, not merely that it runs. No pysal dependency.

Graphs used
-----------
* ``LINE_4``   path  0--1--2--3                  (S0 = 6)
* ``ROOK_2X2`` 2x2 rook grid laid out ``[[0,1],[2,3]]``; neighbour edges are
  (0,1),(0,2),(1,3),(2,3) -> four undirected joins, S0 = 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from hotspots.esda import (
    benjamini_hochberg,
    bivariate_moran_dense,
    join_counts_dense,
    moran_scatter_slope,
    morans_i_dense,
    rook_weights,
    row_standardize,
)

ROOK_2X2 = np.array(
    [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ],
    dtype=float,
)

LINE_4 = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=float,
)


# --------------------------------------------------------------------------- #
# Join counts
# --------------------------------------------------------------------------- #
def test_join_counts_checkerboard_all_bw() -> None:
    """A checkerboard binary field makes every join BW (BB = WW = 0).

    The 2x2 rook grid has four undirected joins: (0,1),(0,2),(1,3),(2,3).
    Field [1,0,0,1] puts a 1 and a 0 at the ends of every one of them, so all
    four joins are mixed.
    """
    bb, ww, bw = join_counts_dense(np.array([1.0, 0.0, 0.0, 1.0]), ROOK_2X2)
    assert (bb, ww, bw) == (0.0, 0.0, 4.0)


def test_join_counts_split_path() -> None:
    """Path 0--1--2--3 with [1,1,0,0]: one BB, one BW, one WW (hand-derived).

    Edges (0,1)=1-1=BB, (1,2)=1-0=BW, (2,3)=0-0=WW.
    """
    bb, ww, bw = join_counts_dense(np.array([1.0, 1.0, 0.0, 0.0]), LINE_4)
    assert (bb, ww, bw) == (1.0, 1.0, 1.0)


def test_join_counts_total_equals_edge_count() -> None:
    """BB + WW + BW always equals the number of undirected joins (S0 / 2)."""
    field = np.array([1.0, 0.0, 1.0, 0.0])
    bb, ww, bw = join_counts_dense(field, ROOK_2X2)
    assert bb + ww + bw == ROOK_2X2.sum() / 2.0


def test_join_counts_non_binary_raises() -> None:
    with pytest.raises(ValueError):
        join_counts_dense(np.array([0.0, 1.0, 2.0, 0.0]), ROOK_2X2)


# --------------------------------------------------------------------------- #
# Bivariate Moran's I
# --------------------------------------------------------------------------- #
def test_bivariate_reduces_to_univariate() -> None:
    """With x == y the bivariate Moran equals the univariate Moran's I."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert bivariate_moran_dense(x, x, LINE_4) == pytest.approx(
        morans_i_dense(x, LINE_4), abs=1e-12
    )


def test_bivariate_opposite_trends_is_negative() -> None:
    """x rising and y falling along the path give a known I_xy = -1/3.

    On LINE_4, x = [0,1,2,3] and y = 3 - x = [3,2,1,0] are exact mirror images,
    so the bivariate Moran is the negative of the univariate I = 1/3.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0, 0.0])
    assert bivariate_moran_dense(x, y, LINE_4) == pytest.approx(-1.0 / 3.0, abs=1e-12)


def test_bivariate_zero_variance_raises() -> None:
    with pytest.raises(ValueError):
        bivariate_moran_dense(
            np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0]), LINE_4
        )


# --------------------------------------------------------------------------- #
# Moran scatter slope identity
# --------------------------------------------------------------------------- #
def test_scatter_slope_equals_row_standardized_moran() -> None:
    """The scatterplot slope equals Moran's I on the row-standardised W.

    On LINE_4 with [0,1,2,3] both equal 2/5 (the row-standardised global I).
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    w_rs = row_standardize(LINE_4)
    slope = moran_scatter_slope(x, LINE_4)
    assert slope == pytest.approx(morans_i_dense(x, w_rs), abs=1e-12)
    assert slope == pytest.approx(2.0 / 5.0, abs=1e-12)


def test_scatter_slope_matches_moran_on_random() -> None:
    """Slope == Moran's I on the same row-standardised W, for random data."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=9)
    a = rng.integers(0, 2, size=(9, 9)).astype(float)
    a = np.triu(a, 1)
    a = a + a.T
    w_rs = row_standardize(a)
    assert moran_scatter_slope(x, a) == pytest.approx(morans_i_dense(x, w_rs), rel=1e-12)


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR
# --------------------------------------------------------------------------- #
def test_bh_known_vector() -> None:
    """Worked BH example.

    p = [0.001, 0.008, 0.039, 0.041, 0.042], m = 5, alpha = 0.05.
    Critical values k/m * alpha = [0.01, 0.02, 0.03, 0.04, 0.05].
    Compare sorted p to crit: 0.001<=0.01 T, 0.008<=0.02 T, 0.039<=0.03 F,
    0.041<=0.04 F, 0.042<=0.05 T. The largest passing rank is k=5, so the
    threshold is p_(5) = 0.042 and every p <= 0.042 is rejected -> all five.
    """
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    reject, thresh = benjamini_hochberg(p, alpha=0.05)
    assert thresh == pytest.approx(0.042, abs=1e-12)
    assert reject.tolist() == [True, True, True, True, True]


def test_bh_rejects_nothing() -> None:
    """When no rank clears its critical value, reject nothing; threshold 0."""
    p = np.array([0.9, 0.8, 0.7])
    reject, thresh = benjamini_hochberg(p, alpha=0.05)
    assert thresh == 0.0
    assert not reject.any()


def test_bh_respects_input_order() -> None:
    """The reject mask is aligned to the input order, not the sorted order."""
    p = np.array([0.042, 0.001, 0.041, 0.008, 0.039])  # shuffled known vector
    reject, thresh = benjamini_hochberg(p, alpha=0.05)
    assert thresh == pytest.approx(0.042, abs=1e-12)
    assert reject.all()


@pytest.mark.parametrize(
    "p, alpha",
    [
        (np.array([]), 0.05),  # empty
        (np.array([0.1, 1.5]), 0.05),  # p outside [0,1]
        (np.array([0.1, 0.2]), 0.0),  # alpha not in (0,1]
    ],
)
def test_bh_invalid_raises(p, alpha) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        benjamini_hochberg(p, alpha=alpha)


# --------------------------------------------------------------------------- #
# Grid weight helpers
# --------------------------------------------------------------------------- #
def test_rook_weights_2x2_matches_reference() -> None:
    """rook_weights(2, 2) reproduces the canonical 2x2 rook adjacency."""
    assert np.array_equal(rook_weights(2, 2), ROOK_2X2)


def test_rook_weights_is_symmetric_zero_diagonal() -> None:
    w = rook_weights(3, 4)
    assert w.shape == (12, 12)
    assert np.array_equal(w, w.T)
    assert np.all(np.diag(w) == 0.0)
    # Interior cell (row 1, col 1) -> index 5 has all four rook neighbours.
    assert w[5].sum() == 4.0
    # A corner (index 0) has exactly two.
    assert w[0].sum() == 2.0


def test_rook_weights_invalid_raises() -> None:
    with pytest.raises(ValueError):
        rook_weights(1, 1)  # only one cell, no neighbours
    with pytest.raises(ValueError):
        rook_weights(0, 5)


def test_row_standardize_rows_sum_to_one() -> None:
    rs = row_standardize(ROOK_2X2)
    assert np.allclose(rs.sum(axis=1), 1.0)


def test_row_standardize_leaves_islands_zero() -> None:
    """An all-zero row (an island) stays zero rather than dividing by zero."""
    w = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rs = row_standardize(w)
    assert np.all(rs[2] == 0.0)
    assert rs[0, 1] == 1.0
