"""Tests for the offline demo (``app.demo``).

These assert the *pure* metrics the demo returns are deterministic and equal the
committed values. They do NOT require matplotlib or Pillow -- ``png_written`` may
be False in a minimal CI environment, so it is not asserted as True.
"""

from __future__ import annotations

import json

import pytest

from app import demo

# Committed numbers for run_demo(seed=0). Derived once from a known-good run;
# regenerating them would mean the synthetic generator changed.
EXPECTED_AREA_KM2 = 79.039038
EXPECTED_NDVI_MEAN = 0.613485
EXPECTED_CACHE_KEY = "eo-explorer:8efeec1e807ae802"


def test_run_demo_metrics_are_deterministic(tmp_path):
    pytest.importorskip("numpy")
    result = demo.run_demo(seed=0, out_dir=str(tmp_path))

    assert result["bbox"] == [10.0, 50.0, 10.1, 50.1]
    assert result["aoi_ok"] is True
    assert result["area_km2"] == EXPECTED_AREA_KM2
    assert result["ndvi_mean"] == EXPECTED_NDVI_MEAN
    assert result["ndvi_valid_fraction"] == 1.0
    assert result["cache_key"] == EXPECTED_CACHE_KEY
    # png_written depends on optional matplotlib/PIL; just assert it is a bool.
    assert isinstance(result["png_written"], bool)


def test_run_demo_writes_summary_json(tmp_path):
    pytest.importorskip("numpy")
    demo.run_demo(seed=0, out_dir=str(tmp_path))
    summary_file = tmp_path / "summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["cache_key"] == EXPECTED_CACHE_KEY
    assert summary["seed"] == 0


def test_run_demo_is_repeatable(tmp_path):
    pytest.importorskip("numpy")
    a = demo.run_demo(seed=0, out_dir=str(tmp_path / "a"))
    b = demo.run_demo(seed=0, out_dir=str(tmp_path / "b"))
    assert a == b


def test_run_demo_seed_changes_metrics(tmp_path):
    pytest.importorskip("numpy")
    a = demo.run_demo(seed=0, out_dir=str(tmp_path / "a"))
    b = demo.run_demo(seed=1, out_dir=str(tmp_path / "b"))
    # AOI geometry is fixed, so area/cache_key match, but the synthetic scene
    # (and therefore the NDVI mean) must differ with a different seed.
    assert a["area_km2"] == b["area_km2"]
    assert a["cache_key"] == b["cache_key"]
    assert a["ndvi_mean"] != b["ndvi_mean"]
