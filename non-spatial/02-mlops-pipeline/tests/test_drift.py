"""Known-answer tests for the PSI / KS drift core.

Every expected value here is hand-derivable on a tiny example, so a green test
proves the statistic is correct rather than merely that it runs. The drift module
has no third-party dependency beyond numpy / pandas, so these always execute.

Worked KS example (``test_ks_hand_value``):

    reference = [1, 2, 3, 4], current = [2, 2, 3, 5].
    Pooled / sorted distinct points and the two ECDFs (fraction <= x):

        x:        1     2     3     4     5
        F_ref:   1/4   2/4   3/4   4/4   4/4
        F_cur:   0/4   2/4   3/4   3/4   4/4
        |gap|:   1/4    0     0    1/4    0

    The maximum gap is 1/4 = 0.25.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlpipe.drift import feature_drift_report, ks_statistic, psi


def test_psi_identical_samples_is_near_zero() -> None:
    """The PSI of a sample against itself is ~0 (well under the 0.1 line)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_two_draws_same_distribution_is_small() -> None:
    """Two independent draws from one distribution sit below the 0.1 threshold."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=5000)
    b = rng.normal(0.0, 1.0, size=5000)
    assert psi(a, b) < 0.1


def test_psi_clear_shift_exceeds_major_threshold() -> None:
    """A clearly shifted sample pushes PSI past the 0.2 major-shift line."""
    rng = np.random.default_rng(2)
    reference = rng.normal(0.0, 1.0, size=5000)
    current = rng.normal(3.0, 1.0, size=5000)  # mean shifted by 3 sigma
    assert psi(reference, current) > 0.2


def test_psi_grows_with_shift() -> None:
    """PSI increases monotonically as the shift grows."""
    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 1.0, size=5000)
    small = psi(reference, rng.normal(0.5, 1.0, size=5000))
    large = psi(reference, rng.normal(2.0, 1.0, size=5000))
    assert large > small


def test_psi_rejects_empty() -> None:
    with pytest.raises(ValueError):
        psi([], [1.0, 2.0])
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [])


def test_ks_identical_is_zero() -> None:
    """KS D of a sample against itself is exactly 0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ks_statistic(x, x) == 0.0


def test_ks_disjoint_support_is_one() -> None:
    """Samples with non-overlapping support give D = 1."""
    reference = np.array([0.0, 0.1, 0.2, 0.3])
    current = np.array([10.0, 10.1, 10.2, 10.3])
    assert ks_statistic(reference, current) == pytest.approx(1.0, abs=1e-12)


def test_ks_hand_value() -> None:
    """Tiny hand-derived pair: max ECDF gap is 0.25 (see module docstring)."""
    reference = [1.0, 2.0, 3.0, 4.0]
    current = [2.0, 2.0, 3.0, 5.0]
    assert ks_statistic(reference, current) == pytest.approx(0.25, abs=1e-12)


def test_ks_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ks_statistic([], [1.0])


def test_feature_drift_report_flags_only_the_shifted_column() -> None:
    """Only the column that moved is flagged drifted; the stable one is not."""
    rng = np.random.default_rng(4)
    n = 4000
    reference = pd.DataFrame(
        {
            "stable": rng.normal(0.0, 1.0, size=n),
            "shifted": rng.normal(0.0, 1.0, size=n),
        }
    )
    current = pd.DataFrame(
        {
            "stable": rng.normal(0.0, 1.0, size=n),
            "shifted": rng.normal(4.0, 1.0, size=n),  # big shift
        }
    )
    report = feature_drift_report(reference, current, psi_threshold=0.2)

    flags = dict(zip(report["feature"], report["drifted"], strict=False))
    assert flags["shifted"] is True
    assert flags["stable"] is False

    summary = report.attrs["summary"]
    assert summary["n_drifted"] == 1
    assert summary["drifted_features"] == ["shifted"]


def test_feature_drift_report_rejects_mismatched_columns() -> None:
    """Different column sets raise ValueError."""
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        feature_drift_report(a, b)


def test_feature_drift_report_rejects_empty() -> None:
    """Empty frames raise ValueError."""
    empty = pd.DataFrame({"x": []})
    full = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        feature_drift_report(empty, full)
