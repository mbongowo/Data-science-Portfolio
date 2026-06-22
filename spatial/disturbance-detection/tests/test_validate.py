"""Detection-rate / false-alarm arithmetic on synthetic event vs detection maps."""

from __future__ import annotations

import numpy as np
import pytest

from disturb.validate import spatial_agreement


def _grid(event_date: str = "2020-09-04"):
    """A 4x4 grid: top-left 2x2 block (4 px) is the event footprint."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    dates = np.full((4, 4), "NaT", dtype="datetime64[D]")
    mag = np.zeros((4, 4), dtype=float)
    return mask, dates, mag, event_date


def test_detection_and_false_alarm_arithmetic():
    mask, dates, mag, ev = _grid()
    # 3 of the 4 event pixels detected, in window, with a real drop.
    for px in [(0, 0), (0, 1), (1, 0)]:
        dates[px] = "2020-09-10"
        mag[px] = -0.4
    # One detection outside the footprint -> one false alarm.
    dates[3, 3] = "2020-09-12"
    mag[3, 3] = -0.5

    res = spatial_agreement(dates, mag, mask, ev, window_days=30,
                            magnitude_threshold=-0.1)

    assert res.n_event_pixels == 4
    assert res.n_outside_pixels == 12
    assert res.n_detected_in_event == 3
    assert res.n_false_alarms == 1
    assert res.detection_rate == pytest.approx(3 / 4)
    assert res.false_alarm_rate == pytest.approx(1 / 12)


def test_out_of_window_detection_is_ignored():
    mask, dates, mag, ev = _grid()
    for px in [(0, 0), (0, 1), (1, 0)]:
        dates[px] = "2020-09-10"
        mag[px] = -0.4
    # Fourth event pixel detected, but months away from the event date.
    dates[1, 1] = "2021-02-01"
    mag[1, 1] = -0.4

    res = spatial_agreement(dates, mag, mask, ev, window_days=30,
                            magnitude_threshold=-0.1)
    assert res.n_detected_in_event == 3
    assert res.detection_rate == pytest.approx(3 / 4)


def test_subthreshold_magnitude_is_ignored():
    mask, dates, mag, ev = _grid()
    # In window and inside footprint, but the drop is too shallow.
    dates[0, 0] = "2020-09-05"
    mag[0, 0] = -0.05  # weaker than magnitude_threshold of -0.1
    res = spatial_agreement(dates, mag, mask, ev, window_days=30,
                            magnitude_threshold=-0.1)
    assert res.n_detected_in_event == 0
    assert res.detection_rate == pytest.approx(0.0)


def test_positive_magnitude_never_counts_as_disturbance():
    mask, dates, mag, ev = _grid()
    dates[0, 0] = "2020-09-05"
    mag[0, 0] = 0.4  # a rise, not a loss
    res = spatial_agreement(dates, mag, mask, ev, window_days=30,
                            magnitude_threshold=-0.1)
    assert res.n_detected_in_event == 0


def test_window_boundary_is_inclusive():
    mask, dates, mag, ev = _grid()
    # Exactly window_days after the event date counts; one day later does not.
    dates[0, 0] = "2020-10-04"  # +30 days
    mag[0, 0] = -0.4
    dates[0, 1] = "2020-10-05"  # +31 days
    mag[0, 1] = -0.4
    res = spatial_agreement(dates, mag, mask, ev, window_days=30,
                            magnitude_threshold=-0.1)
    assert res.n_detected_in_event == 1


def test_perfect_detection_no_false_alarms():
    mask, dates, mag, ev = _grid()
    dates[mask] = "2020-09-08"
    mag[mask] = -0.5
    res = spatial_agreement(dates, mag, mask, ev, window_days=30,
                            magnitude_threshold=-0.1)
    assert res.detection_rate == pytest.approx(1.0)
    assert res.false_alarm_rate == pytest.approx(0.0)


def test_mismatched_shapes_raise():
    mask, dates, mag, ev = _grid()
    with pytest.raises(ValueError):
        spatial_agreement(dates[:, :3], mag, mask, ev)


def test_summary_string_reports_rates():
    mask, dates, mag, ev = _grid()
    dates[mask] = "2020-09-08"
    mag[mask] = -0.5
    res = spatial_agreement(dates, mag, mask, ev)
    text = str(res)
    assert "detection rate" in text
    assert "false-alarm" in text
