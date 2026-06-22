"""Tests for the offline demo: deterministic metrics + planted-anomaly recovery.

The metric values below are pinned for ``seed=0`` so a regression in the index
or anomaly core, or in the synthetic-cube construction, fails loudly.
"""

from __future__ import annotations

import json

import numpy as np

from eo_monitor.demo import GRID, run_demo


def test_run_demo_metrics_pinned_seed0(tmp_path) -> None:
    m = run_demo(seed=0, out_dir=tmp_path)

    assert m["seed"] == 0
    assert m["grid_shape"] == list(GRID)
    # Healthy NDVI level for the synthesised vegetation (high NIR, low red).
    assert m["mean_ndvi"] == 0.683053
    assert m["mean_ndwi"] == -0.582437
    assert m["mean_ndmi"] == 0.30189
    # 80 planted pixels plus a handful of baseline-noise tail pixels.
    assert m["planted_pixel_count"] == 80
    assert m["anomaly_pixel_count"] == 93
    assert m["anomaly_fraction"] == 0.161458
    assert m["max_abs_z"] == 42.120155


def test_planted_anomaly_fully_recovered(tmp_path) -> None:
    # The |z| > 2 loss mask must recover the entire planted block (recall 1.0).
    m = run_demo(seed=0, out_dir=tmp_path)
    assert m["planted_region_recovered"] == 1.0


def test_demo_is_deterministic(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b


def test_demo_writes_artifacts(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)
    summary = tmp_path / "summary.json"
    ndvi_map = tmp_path / "ndvi_demo.npy"
    zmap = tmp_path / "ndvi_zscore.npy"
    assert summary.exists() and ndvi_map.exists() and zmap.exists()

    # summary.json round-trips and matches the returned dict.
    loaded = json.loads(summary.read_text(encoding="utf-8"))
    assert loaded["anomaly_pixel_count"] == 93

    # The saved z-score field has a strong negative tail (the planted loss).
    z = np.load(zmap)
    assert np.nanmin(z) < -2.0
    assert z.shape == GRID


def test_demo_seed_changes_noise_not_recovery(tmp_path) -> None:
    # A different seed reshuffles the noise but the planted block is still found.
    m1 = run_demo(seed=1, out_dir=tmp_path / "s1")
    assert m1["planted_region_recovered"] == 1.0
    assert m1["planted_pixel_count"] == 80
