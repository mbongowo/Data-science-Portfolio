"""Known-answer tests for the event-count matrix.

Hand-worked example over three templates (0, 1, 2):

    session "a" : [0, 0, 1]  -> counts [2, 1, 0]
    session "b" : [2, 1, 1]  -> counts [0, 2, 1]

Row order follows the dict insertion order; column j counts template j.
"""

from __future__ import annotations

import numpy as np
import pytest

from loganomaly.features import event_count_matrix


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
