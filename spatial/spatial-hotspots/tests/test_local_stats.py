"""Known-answer tests for the pure-numpy local-statistics reference layer.

These cover Geary's C, Local Moran's I (LISA), the LISA quadrant labels, and
the standardised Getis-Ord Gi*. As in ``test_esda.py`` the expected values are
hand-derived on tiny graphs and cross-checked with exact rational arithmetic
(``fractions.Fraction``), so a green test proves the estimator is correct rather
than merely that it runs. None of these tests import pysal.

Graphs used
-----------
* ``LINE_4``   path  0--1--2--3      (S0 = 6)
* ``LINE_5``   path  0--1--2--3--4   (S0 = 8)
* ``ROOK_2X2`` 2x2 rook grid laid out as ``[[0,1],[2,3]]`` (S0 = 8)

Worked derivations (verified independently with Fraction):

Geary's C, ``C = (n-1)/(2 S0) * sum_ij w_ij (x_i - x_j)^2 / sum_i (x_i-xbar)^2``.

* LINE_4 with values [0,1,2,3]: S0 = 6, the three unit-gap edges contribute
  ``2 * (1 + 1 + 1) = 6`` to the numerator, denominator ``z'z = 5``, so
  ``C = 3/(2*6) * 6/5 = 3/10`` (clustering, C < 1).
* ROOK_2X2 checkerboard [1,-1,-1,1]: S0 = 8, each of the eight directed edges
  has squared gap 4, numerator 32, denominator 4, so
  ``C = 3/16 * 32/4 = 3/2`` (dispersion, C > 1).

Local Moran, ``I_i = (z_i / m2) * (W_row z)_i`` with row-standardised W.

* LINE_4 with [0,1,2,3]: z = [-3/2,-1/2,1/2,3/2], m2 = 5/4, lags
  [-1/2,-1/2,1/2,1/2], so I = [3/5, 1/5, 1/5, 3/5] and mean(I) = 2/5 equals the
  row-standardised global Moran's I.
* LINE_5 with [0,0,0,2,1] gives all four quadrants: labels
  [LL, LL, LH, HL, HH] and I = [9/16, 9/16, -3/8, -7/32, 7/8].

Getis-Ord Gi* (star, binary weights, self included).

* ROOK_2X2 with one hot corner [10,1,1,1]: Xbar = 13/4, S = sqrt(243/16). Every
  3-cell window has variance term 1, so nodes 0,1,2 score 1/sqrt(3) and node 3
  scores -sqrt(3).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from hotspots.esda import (
    gearys_c_dense,
    getis_ord_g_star_dense,
    lisa_quadrants,
    local_moran_dense,
    morans_i_dense,
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

LINE_5 = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ],
    dtype=float,
)


# --------------------------------------------------------------------------- #
# Geary's C
# --------------------------------------------------------------------------- #
def test_geary_line_monotone_is_three_tenths() -> None:
    """Increasing values along a path cluster => C = 3/10 (hand-derived)."""
    values = np.array([0.0, 1.0, 2.0, 3.0])
    assert gearys_c_dense(values, LINE_4) == pytest.approx(3.0 / 10.0, abs=1e-12)


def test_geary_checkerboard_is_three_halves() -> None:
    """Checkerboard disperses => C = 3/2 (hand-derived), the C > 1 regime."""
    values = np.array([1.0, -1.0, -1.0, 1.0])
    assert gearys_c_dense(values, ROOK_2X2) == pytest.approx(3.0 / 2.0, abs=1e-12)


def test_geary_translation_invariant() -> None:
    """Geary's C uses centred variance and squared gaps; a shift cannot move it."""
    a = gearys_c_dense(np.array([0.0, 1.0, 2.0, 3.0]), LINE_4)
    b = gearys_c_dense(np.array([100.0, 101.0, 102.0, 103.0]), LINE_4)
    assert a == pytest.approx(b, abs=1e-12)


def test_geary_matches_fraction_reference() -> None:
    """Cross-check against an exact rational reimplementation on random data."""
    rng = np.random.default_rng(7)
    x = rng.integers(-5, 6, size=9)
    a = rng.integers(0, 2, size=(9, 9))
    a = np.triu(a, 1)
    a = a + a.T  # symmetric binary, zero diagonal

    xf = [Fraction(int(v)) for v in x]
    n = len(xf)
    s0 = Fraction(int(a.sum()))
    xbar = sum(xf, Fraction(0)) / n
    num = sum(
        Fraction(int(a[i, j])) * (xf[i] - xf[j]) ** 2
        for i in range(n)
        for j in range(n)
    )
    den = sum((xi - xbar) ** 2 for xi in xf)
    expected = Fraction(n - 1, 1) * num / (2 * s0 * den)

    assert gearys_c_dense(x.astype(float), a.astype(float)) == pytest.approx(
        float(expected), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# Local Moran (LISA)
# --------------------------------------------------------------------------- #
def test_local_moran_line_values() -> None:
    """Per-node LISA on the monotone path matches the hand-derived vector."""
    values = np.array([0.0, 1.0, 2.0, 3.0])
    got = local_moran_dense(values, LINE_4)
    expected = np.array([3 / 5, 1 / 5, 1 / 5, 3 / 5])
    assert got == pytest.approx(expected, abs=1e-12)


def test_local_moran_mean_equals_global() -> None:
    """With a row-standardised W, mean(I_i) equals the global Moran's I."""
    values = np.array([0.0, 1.0, 2.0, 3.0])
    local = local_moran_dense(values, LINE_4)
    # Global Moran's I on a row-standardised copy of the same graph.
    w_rs = LINE_4 / LINE_4.sum(axis=1, keepdims=True)
    glob = morans_i_dense(values, w_rs)
    assert local.mean() == pytest.approx(glob, abs=1e-12)
    assert local.mean() == pytest.approx(2.0 / 5.0, abs=1e-12)


def test_local_moran_five_node_values() -> None:
    """The four-quadrant example reproduces the exact rational LISA vector."""
    values = np.array([0.0, 0.0, 0.0, 2.0, 1.0])
    got = local_moran_dense(values, LINE_5)
    expected = np.array([9 / 16, 9 / 16, -3 / 8, -7 / 32, 7 / 8])
    assert got == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# LISA quadrant labels
# --------------------------------------------------------------------------- #
def test_lisa_quadrants_all_four_present() -> None:
    """A constructed path graph yields every quadrant in a known order."""
    values = np.array([0.0, 0.0, 0.0, 2.0, 1.0])
    labels = lisa_quadrants(values, LINE_5)
    assert list(labels) == ["LL", "LL", "LH", "HL", "HH"]


def test_lisa_quadrants_monotone_path() -> None:
    """On the monotone path the two halves are LL (low end) and HH (high end)."""
    values = np.array([0.0, 1.0, 2.0, 3.0])
    labels = lisa_quadrants(values, LINE_4)
    assert list(labels) == ["LL", "LL", "HH", "HH"]


def test_lisa_quadrant_sign_consistency() -> None:
    """A HH/LL label requires I_i > 0; an HL/LH label requires I_i < 0."""
    values = np.array([0.0, 0.0, 0.0, 2.0, 1.0])
    labels = lisa_quadrants(values, LINE_5)
    local = local_moran_dense(values, LINE_5)
    for lab, ii in zip(labels, local):
        if lab in ("HH", "LL"):
            assert ii > 0
        elif lab in ("HL", "LH"):
            assert ii < 0


# --------------------------------------------------------------------------- #
# Getis-Ord Gi*
# --------------------------------------------------------------------------- #
def test_gi_star_hot_corner() -> None:
    """One hot corner on the 2x2 rook gives 1/sqrt(3) and -sqrt(3) (derived)."""
    values = np.array([10.0, 1.0, 1.0, 1.0])
    got = getis_ord_g_star_dense(values, ROOK_2X2)
    expected = np.array(
        [
            1.0 / np.sqrt(3.0),
            1.0 / np.sqrt(3.0),
            1.0 / np.sqrt(3.0),
            -np.sqrt(3.0),
        ]
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_gi_star_hot_spot_is_positive() -> None:
    """The unit holding the high value and its neighbours score positive."""
    values = np.array([10.0, 1.0, 1.0, 1.0])
    z = getis_ord_g_star_dense(values, ROOK_2X2)
    assert z[0] > 0 and z[1] > 0 and z[2] > 0
    assert z[3] < 0


def test_gi_star_matches_fraction_reference() -> None:
    """Cross-check the numerator and variance term with exact arithmetic."""
    values = np.array([10.0, 1.0, 1.0, 1.0])
    xf = [Fraction(int(v)) for v in values]
    n = len(xf)
    # star weights: rook adjacency with unit diagonal
    w = ROOK_2X2.astype(int).tolist()
    for i in range(n):
        w[i][i] = 1
    xbar = sum(xf, Fraction(0)) / n
    s2 = sum(v * v for v in xf) / n - xbar * xbar  # = 243/16
    got = getis_ord_g_star_dense(values, ROOK_2X2)
    s = float(s2) ** 0.5
    for i in range(n):
        wsum = sum(w[i])
        wsq = sum(v * v for v in w[i])
        numer = sum(Fraction(w[i][j]) * xf[j] for j in range(n)) - xbar * wsum
        var = (Fraction(n) * wsq - wsum * wsum) / (n - 1)
        expected = float(numer) / (s * float(var) ** 0.5)
        assert got[i] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn",
    [gearys_c_dense, local_moran_dense],
)
def test_zero_variance_raises(fn) -> None:  # type: ignore[no-untyped-def]
    constant = np.array([3.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError):
        fn(constant, LINE_4)


def test_gi_star_zero_variance_raises() -> None:
    with pytest.raises(ValueError):
        getis_ord_g_star_dense(np.array([3.0, 3.0, 3.0, 3.0]), ROOK_2X2)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        gearys_c_dense(np.array([1.0, 2.0]), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        local_moran_dense(np.array([1.0, 2.0]), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        getis_ord_g_star_dense(np.array([1.0, 2.0, 3.0]), np.zeros((2, 2)))
