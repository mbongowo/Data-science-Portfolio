"""Tests for the dependency-free synthetic flood-mapping demo.

The demo synthesises a before/after SAR backscatter pair with a permanent river
and a planted flood block, then drives the pure-numpy core. With a fixed seed the
metrics are deterministic, so we pin the headline numbers quoted in the README and
confirm the mapper recovers the planted flood. These tests need only numpy +
stdlib.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from floodmap.demo import run_demo


def test_demo_metrics_are_pinned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Seed=0 yields the exact metrics quoted in the README."""
    out = run_demo(seed=0, out_dir=tmp_path)

    assert out["raster"] == [100, 100]
    assert out["pixel_size_m"] == pytest.approx(10.0)
    assert out["planted_flood_hectares"] == pytest.approx(12.0, abs=1e-9)
    assert out["pre_water_threshold_db"] == pytest.approx(
        PINNED["pre_water_threshold_db"], abs=1e-9
    )
    assert out["post_water_threshold_db"] == pytest.approx(
        PINNED["post_water_threshold_db"], abs=1e-9
    )
    assert out["flooded_hectares"] == pytest.approx(
        PINNED["flooded_hectares"], abs=1e-9
    )
    assert out["permanent_water_hectares"] == pytest.approx(
        PINNED["permanent_water_hectares"], abs=1e-9
    )


def test_demo_recovers_planted_flood(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The planted flood is recovered and produces real flooded hectares."""
    out = run_demo(seed=0, out_dir=tmp_path)
    assert out["flooded_hectares"] > 0
    assert out["planted_flood_recovered"] >= 0.8


def test_demo_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The demo writes flood_stats.json and flood_mask.npy, round-tripping."""
    out = run_demo(seed=0, out_dir=tmp_path)

    stats_path = tmp_path / "flood_stats.json"
    mask_path = tmp_path / "flood_mask.npy"
    assert stats_path.exists()
    assert mask_path.exists()

    with open(stats_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["flooded_hectares"] == out["flooded_hectares"]

    mask = np.load(mask_path)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    # flooded pixels in the mask match the reported count.
    assert int(mask.sum()) == out["flooded_pixels"]


def test_demo_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Same seed -> identical metrics on a second run."""
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b


# Pinned metrics for seed=0 (see README "Result first").
PINNED = {
    "pre_water_threshold_db": -13.476083749603164,
    "post_water_threshold_db": -13.358269377688497,
    "flooded_hectares": 12.1,
    "permanent_water_hectares": 4.01,
}
