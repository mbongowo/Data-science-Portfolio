"""Smoke + pinned-metric tests for the seeded demo."""

from __future__ import annotations

import json

from clinicaccess.demo import run_demo


def test_run_demo_writes_artifacts_and_pins_metrics(tmp_path):
    m = run_demo(seed=0, out_dir=tmp_path)

    assert m["n_places"] == 200
    assert m["n_facilities"] == 20

    # Shares are valid probabilities.
    for key in ("share_within_5km", "share_within_10km", "share_beyond_25km"):
        assert 0.0 <= m[key] <= 1.0

    # The farthest place is at least as far as the median place.
    assert m["farthest_place_km"] >= m["median_nearest_km"]
    assert m["mean_nearest_km"] > 0.0

    # Artifacts exist and the summary round-trips.
    assert (tmp_path / "places_access.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_places"] == 200
    assert len(summary["farthest_places"]) == 10


def test_run_demo_is_deterministic(tmp_path):
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
