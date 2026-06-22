"""Event-count features: turn template-id sequences into a session-by-event matrix.

After templating, each session (an HDFS block, say) is a sequence of template
ids. The standard feature representation for log anomaly detection is the
**event-count matrix**: one row per session, one column per template, each cell
the number of times that template fired in that session. PCA, z-score, and
IsolationForest detectors all operate on this matrix.

Pure numpy/stdlib; no third-party dependency beyond numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray


def event_count_matrix(
    session_to_template_ids: Mapping[object, Iterable[int]],
    n_templates: int,
) -> NDArray[np.float64]:
    """Build an event-count matrix from per-session template-id sequences.

    Parameters
    ----------
    session_to_template_ids:
        Mapping from a session key to its sequence of template ids. Iteration
        order of the mapping fixes the row order of the output (dicts preserve
        insertion order in Python 3.7+).
    n_templates:
        Total number of distinct templates; the number of columns. Every id in
        the sequences must be in ``range(n_templates)``.

    Returns
    -------
    numpy.ndarray
        A ``(n_sessions, n_templates)`` float array. Cell ``(i, j)`` is the
        number of times template ``j`` occurred in session ``i``.

    Raises
    ------
    ValueError
        If ``n_templates`` is negative, or a template id falls outside
        ``range(n_templates)``.

    Examples
    --------
    Two sessions over three templates:

    >>> m = event_count_matrix({"a": [0, 0, 1], "b": [2, 1, 1]}, 3)
    >>> m.tolist()
    [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]
    """
    if n_templates < 0:
        raise ValueError("n_templates must be non-negative.")

    n_sessions = len(session_to_template_ids)
    matrix = np.zeros((n_sessions, n_templates), dtype=float)

    for row, ids in enumerate(session_to_template_ids.values()):
        for tid in ids:
            if not 0 <= tid < n_templates:
                raise ValueError(f"Template id {tid} out of range [0, {n_templates}).")
            matrix[row, tid] += 1.0
    return matrix
