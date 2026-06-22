"""Known-answer tests for the event-count matrix.

Hand-worked example over three templates (0, 1, 2):

    session "a" : [0, 0, 1]  -> counts [2, 1, 0]
    session "b" : [2, 1, 1]  -> counts [0, 2, 1]

Row order follows the dict insertion order; column j counts template j.
"""

from __future__ import annotations

import numpy as np
import pytest

from loganomaly.features import (
    count_invariants,
    event_count_matrix,
    session_rarity,
    template_idf,
)


def test_event_count_matrix_known() -> None:
    """The worked two-session example produces the hand-derived matrix."""
    sessions = {"a": [0, 0, 1], "b": [2, 1, 1]}
    matrix = event_count_matrix(sessions, n_templates=3)
    expected = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    assert np.array_equal(matrix, expected)
    assert matrix.shape == (2, 3)


def test_event_count_matrix_empty_session() -> None:
    """A session with no events is an all-zero row."""
    matrix = event_count_matrix({"a": [], "b": [1]}, n_templates=2)
    assert np.array_equal(matrix, np.array([[0.0, 0.0], [0.0, 1.0]]))


def test_event_count_matrix_row_order_follows_insertion() -> None:
    """Output rows are in mapping (insertion) order."""
    matrix = event_count_matrix({"second": [1], "first": [0]}, n_templates=2)
    assert np.array_equal(matrix, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_event_count_matrix_rejects_out_of_range_id() -> None:
    """A template id outside range raises."""
    with pytest.raises(ValueError):
        event_count_matrix({"a": [3]}, n_templates=2)


# --- Template IDF rarity ----------------------------------------------------
#
# Hand-worked example, N = 4 sessions, n_templates = 3, smoothed IDF
#     idf_j = ln((1 + N) / (1 + df_j)) + 1.
#
#   template 0 appears in all 4 sessions -> df=4 -> ln(5/5)+1 = 1.0
#   template 1 appears in 2 sessions     -> df=2 -> ln(5/3)+1
#   template 2 appears in 1 session      -> df=1 -> ln(5/2)+1
# Rarer template => larger weight, and a template in every session stays at 1.0.

IDF_SESSIONS = [
    [0, 0, 1],  # templates {0, 1}
    [0, 1],  # templates {0, 1}
    [0, 2],  # templates {0, 2}
    [0],  # templates {0}
]


def test_template_idf_known_answer() -> None:
    """IDF matches the smoothed log formula; a universal template stays at 1.0."""
    idf = template_idf(IDF_SESSIONS, n_templates=3)
    expected = np.array(
        [
            np.log(5.0 / 5.0) + 1.0,  # df 4
            np.log(5.0 / 3.0) + 1.0,  # df 2
            np.log(5.0 / 2.0) + 1.0,  # df 1
        ]
    )
    assert np.allclose(idf, expected)
    assert idf[0] == pytest.approx(1.0)  # appears everywhere
    assert idf[2] > idf[1] > idf[0]  # rarer is heavier


def test_template_idf_counts_session_once() -> None:
    """Repeated templates within a session count once toward document frequency."""
    # template 0 fires 5x in the only session -> df 1, not 5.
    idf = template_idf([[0, 0, 0, 0, 0]], n_templates=1)
    assert idf[0] == pytest.approx(np.log(2.0 / 2.0) + 1.0)  # = 1.0


def test_template_idf_unseen_template_no_div_by_zero() -> None:
    """A template that never appears (df 0) gets a finite, max weight."""
    idf = template_idf([[0], [0]], n_templates=2)
    assert np.all(np.isfinite(idf))
    assert idf[1] == pytest.approx(np.log(3.0 / 1.0) + 1.0)


def test_session_rarity_known_answer() -> None:
    """Rarity = event counts dot IDF weights, per session."""
    idf = np.array([1.0, 2.0, 5.0])
    counts = np.array(
        [
            [3.0, 0.0, 0.0],  # only common template -> 3.0
            [0.0, 0.0, 1.0],  # one rare template    -> 5.0
            [1.0, 1.0, 1.0],  # one of each          -> 8.0
        ]
    )
    scores = session_rarity(counts, idf)
    assert scores.tolist() == [3.0, 5.0, 8.0]
    assert int(np.argmax(scores)) == 2


def test_session_rarity_shape_mismatch_raises() -> None:
    """Column count must match the IDF length."""
    with pytest.raises(ValueError):
        session_rarity(np.zeros((2, 3)), np.array([1.0, 2.0]))


# --- Invariants band check --------------------------------------------------


def test_count_invariants_known_band() -> None:
    """A row that overshoots a column's upper band is flagged; in-band rows are not.

    Both columns are constant at 1 across the first four rows, so each column's
    [0.05, 0.95] band collapses to [1, 1]. The last session fires template 0
    five times -> above its band -> flagged; nothing else leaves any band.
    """
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [5.0, 1.0],  # template 0 overshoots -> flagged (index 4)
        ]
    )
    mask = count_invariants(X)
    assert mask[4]
    assert mask.tolist() == [False, False, False, False, True]


def test_count_invariants_all_in_band() -> None:
    """When every session shares the same counts, nothing violates its band."""
    X = np.full((5, 3), 2.0)
    assert not count_invariants(X).any()


def test_count_invariants_empty_matrix() -> None:
    """A matrix with no columns flags nothing (and matches the row count)."""
    mask = count_invariants(np.zeros((4, 0)))
    assert mask.tolist() == [False, False, False, False]


def test_count_invariants_bad_quantiles_raise() -> None:
    """lower must not exceed upper, both in [0, 1]."""
    with pytest.raises(ValueError):
        count_invariants(np.zeros((3, 2)), lower_quantile=0.9, upper_quantile=0.1)
