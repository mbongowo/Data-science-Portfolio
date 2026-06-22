"""Tests for the dependency-free synthetic analytics demo.

The demo lays out a 5x5 block of building footprints plus one large field on a
120x120 raster and drives the pure-numpy core. With a fixed seed the metrics are
deterministic, so we pin the headline numbers quoted in the README and confirm
the reconstruction IoU sanity check is exactly 1.0. These tests need only numpy
+ stdlib.
"""

from __future__ import annotations

import json

import pytest

from samgeo_post.demo import run_demo


def test_demo_metrics_are_pinned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Seed=0 yields the exact metrics quoted in the README."""
    out = run_demo(seed=0, out_dir=tmp_path)

    assert out["n_objects"] == 26
    assert out["n_buildings"] == 25
    assert out["mean_building_footprint_m2"] == pytest.approx(26.77, abs=1e-9)
    assert out["total_building_area_m2"] == pytest.approx(669.25, abs=1e-9)
    assert out["field_area_hectares"] == pytest.approx(0.0575, abs=1e-9)
    assert out["pixel_size_m"] == pytest.approx(0.5)


def test_demo_reconstruction_iou_is_one(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The labelling round-trips the input mask exactly."""
    out = run_demo(seed=0, out_dir=tmp_path)
    assert out["reconstruction_iou"] == pytest.approx(1.0, abs=1e-12)


def test_demo_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The demo writes summary.json and region_props.csv, and they round-trip."""
    out = run_demo(seed=0, out_dir=tmp_path)

    summary_path = tmp_path / "summary.json"
    props_path = tmp_path / "region_props.csv"
    assert summary_path.exists()
    assert props_path.exists()

    with open(summary_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["n_buildings"] == out["n_buildings"]

    # Header row plus one row per labelled object (26 objects).
    lines = props_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 26 + 1


def test_demo_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Same seed -> identical metrics on a second run."""
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
