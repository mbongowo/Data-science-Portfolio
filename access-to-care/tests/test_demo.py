"""Pin the seeded demo's headline metrics and artifact invariants.

The demo synthesizes a small fixed network, so for ``seed=0`` the numbers are
deterministic and can be asserted exactly. Runs pure Python/pandas; no geo stack.
"""

from __future__ import annotations

import json

import pytest

from access.demo import THRESHOLDS_MIN, run_demo


def test_run_demo_returns_pinned_metrics(tmp_path) -> None:
    out = run_demo(seed=0, out_dir=tmp_path)
    assert out["n_nodes"] == 144
    assert out["n_facilities"] == 3
    assert out["population_total"] == pytest.approx(107695.8, abs=0.1)
    assert out["share_within_30min"] == pytest.approx(0.10008, abs=1e-4)
    assert out["share_within_60min"] == pytest.approx(0.35074, abs=1e-4)
    assert out["pop_unreachable"] == pytest.approx(0.0, abs=0.1)


def test_run_demo_shares_in_unit_interval(tmp_path) -> None:
    out = run_demo(seed=0, out_dir=tmp_path)
    for key in ("share_within_30min", "share_within_60min"):
        assert 0.0 <= out[key] <= 1.0


def test_run_demo_is_deterministic(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b


def test_run_demo_writes_artifacts(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)
    demand_csv = tmp_path / "demand.csv"
    summary_json = tmp_path / "summary.json"
    assert demand_csv.exists()
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["seed"] == 0
    assert summary["thresholds_min"] == THRESHOLDS_MIN


def test_run_demo_bands_sum_to_population_total(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    bands = summary["coverage_bands"]
    total = bands["population_total"]
    band_sum = (
        bands["pop_band_0_30min"]
        + bands["pop_band_30_60min"]
        + bands["pop_band_60_120min"]
        + bands["pop_band_120min_plus"]
        + bands["pop_unreachable"]
    )
    assert band_sum == pytest.approx(total)
