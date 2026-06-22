"""Known-answer tests for the pure-numpy Moran's I reference implementation.

These tests deliberately use *hand-computed* expected values on tiny graphs so
that a green test proves the estimator is correct, not merely that it runs.
They have **no third-party (pysal) dependency**, so they always execute.

Reference estimator:

    I = (n / S0) * (z' W z) / (z' z),   z = x - mean(x),  S0 = sum(W)

Worked examples used below:

1. 4-node *line* graph  0--1--2--3  with values [0,1,2,3].
   Edges: (0,1),(1,2),(2,3) symmetric => S0 = 6, n = 4.
   mean = 1.5, z = [-1.5,-0.5,0.5,1.5], z'z = 5.
   z'Wz = 2 * [z0 z1 + z1 z2 + z2 z3]
        = 2 * [(-1.5)(-0.5) + (-0.5)(0.5) + (0.5)(1.5)]
        = 2 * [0.75 - 0.25 + 0.75] = 2 * 1.25 = 2.5
   I = (4/6) * (2.5 / 5) = (2/3) * 0.5 = 1/3.   (strong positive)

2. 2x2 rook grid with a checkerboard [1,-1,-1,1] => I = -1 exactly
   (perfect negative autocorrelation / dispersion).

3. Same grid, symmetric smooth field [0,1,1,2] => I = 0 exactly
   (the high-low contributions cancel).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotspots.esda import expected_morans_i, morans_i_dense

# Rook contiguity for a 2x2 grid laid out as
#   0 1
#   2 3
# (corners are NOT adjacent under rook)
ROOK_2X2 = np.array(
    [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ],
    dtype=float,
)

# Path / line graph 0--1--2--3
LINE_4 = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=float,
)


def test_line_graph_monotone_is_one_third() -> None:
    """Increasing values along a path => I = 1/3 (hand-derived)."""
    values = np.array([0.0, 1.0, 2.0, 3.0])
    result = morans_i_dense(values, LINE_4)
    assert result == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_line_graph_is_translation_invariant() -> None:
    """Adding a constant must not change Moran's I (uses centred values)."""
    a = morans_i_dense(np.array([0.0, 1.0, 2.0, 3.0]), LINE_4)
    b = morans_i_dense(np.array([10.0, 11.0, 12.0, 13.0]), LINE_4)
    assert a == pytest.approx(b, abs=1e-12)


def test_checkerboard_is_perfect_negative() -> None:
    """Checkerboard pattern on a 2x2 rook grid => I = -1 exactly."""
    values = np.array([1.0, -1.0, -1.0, 1.0])
    result = morans_i_dense(values, ROOK_2X2)
    assert result == pytest.approx(-1.0, abs=1e-12)


def test_symmetric_field_is_zero() -> None:
    """A balanced smooth field on the 2x2 grid yields exactly I = 0."""
    values = np.array([0.0, 1.0, 1.0, 2.0])
    result = morans_i_dense(values, ROOK_2X2)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_diagonal_is_ignored() -> None:
    """Self-weights on the diagonal must not affect the statistic."""
    w_with_diag = LINE_4.copy()
    np.fill_diagonal(w_with_diag, 5.0)
    values = np.array([0.0, 1.0, 2.0, 3.0])
    assert morans_i_dense(values, w_with_diag) == pytest.approx(
        1.0 / 3.0, abs=1e-12
    )


def test_expected_value_under_null() -> None:
    """E[I] = -1/(n-1)."""
    assert expected_morans_i(4) == pytest.approx(-1.0 / 3.0, abs=1e-12)
    assert expected_morans_i(101) == pytest.approx(-0.01, abs=1e-12)


@pytest.mark.parametrize(
    "values, w, exc",
    [
        (np.array([1.0]), np.array([[0.0]]), ValueError),  # n < 2
        (np.array([1.0, 2.0]), np.zeros((3, 3)), ValueError),  # shape mismatch
        (np.array([1.0, 2.0]), np.zeros((2, 2)), ValueError),  # S0 == 0
        (np.array([2.0, 2.0]), np.array([[0.0, 1.0], [1.0, 0.0]]), ValueError),
    ],
)
def test_invalid_inputs_raise(values, w, exc) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(exc):
        morans_i_dense(values, w)


def test_matches_naive_quadratic_form() -> None:
    """Cross-check against an independent O(n^2) reimplementation."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=12)
    # random symmetric 0/1 adjacency, zero diagonal
    a = rng.integers(0, 2, size=(12, 12)).astype(float)
    a = np.triu(a, 1)
    a = a + a.T
    z = x - x.mean()
    n = x.size
    s0 = a.sum()
    num = sum(z[i] * a[i, j] * z[j] for i in range(n) for j in range(n))
    den = float(z @ z)
    expected = (n / s0) * (num / den)
    assert morans_i_dense(x, a) == pytest.approx(expected, rel=1e-12)
    assert not math.isnan(morans_i_dense(x, a))
