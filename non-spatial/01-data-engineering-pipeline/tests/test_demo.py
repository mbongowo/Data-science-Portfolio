"""Pin the seeded demo's headline metrics and artifacts."""

from __future__ import annotations

import json

from weatherpipe.demo import N_PLANTED_BAD, run_demo


def test_run_demo_pins_metrics(tmp_path) -> None:
    out = run_demo(seed=0, out_dir=str(tmp_path))

    assert out["n_stations"] == 5
    assert out["n_records"] == 1823
    # Exactly the planted bad rows are rejected.
    assert out["n_rejected"] == N_PLANTED_BAD == 3
    assert 0.0 < out["pct_valid"] <= 1.0

    hottest = out["hottest_station_month"]
    assert hottest["station"] == "Garoua"
    assert hottest["year"] == 2023
    assert hottest["month"] == 6
    assert hottest["tmean"] == 35.74

    assert out["wettest_month_total_precip_mm"] == 673.23


def test_run_demo_writes_artifacts(tmp_path) -> None:
    run_demo(seed=0, out_dir=str(tmp_path))

    monthly = tmp_path / "monthly_summary.csv"
    report = tmp_path / "validation_report.json"
    assert monthly.exists()
    assert report.exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["n_rejected"] == 3
    assert payload["rejected"]["tmin_gt_tmax"] == 1
    assert payload["rejected"]["range"] == 1
    assert payload["rejected"]["duplicate"] == 1


def test_run_demo_is_deterministic(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=str(tmp_path / "a"))
    b = run_demo(seed=0, out_dir=str(tmp_path / "b"))
    assert a == b
