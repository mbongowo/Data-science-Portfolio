"""Committed-value tests for the end-to-end synthetic demo.

These pin the *real* numbers the README quotes. ``run_demo(seed=0)`` is fully
deterministic (numpy ``default_rng(0)``), so the counts below are exact and
regenerate the documented metrics. The clean synthetic marts must pass every
generic test; the planted-defect contrast run must fail some.
"""

from __future__ import annotations

import json

from dwh.demo import run_demo


def test_run_demo_committed_metrics(tmp_path) -> None:
    summary = run_demo(seed=0, out_dir=str(tmp_path))

    # Dataset shape (deterministic for seed=0).
    assert summary["num_titles"] == 200
    assert summary["num_ratings"] == 160

    # The full generic-test suite over the clean marts: all pass.
    assert summary["num_tests"] == 13
    assert summary["num_passed"] == 13
    assert summary["num_failed"] == 0


def test_clean_data_passes_every_test(tmp_path) -> None:
    summary = run_demo(seed=0, out_dir=str(tmp_path))
    assert summary["num_passed"] == summary["num_tests"]
    for kind, counts in summary["breakdown"].items():
        assert counts["passed"] == counts["total"], kind


def test_breakdown_covers_all_four_generic_tests(tmp_path) -> None:
    summary = run_demo(seed=0, out_dir=str(tmp_path))
    assert set(summary["breakdown"]) == {
        "not_null",
        "unique",
        "accepted_values",
        "relationships",
    }


def test_dirty_contrast_run_catches_defects(tmp_path) -> None:
    summary = run_demo(seed=0, out_dir=str(tmp_path))
    assert summary["dirty_num_tests"] == 13
    assert summary["dirty_num_failed"] == 5


def test_artifacts_written(tmp_path) -> None:
    run_demo(seed=0, out_dir=str(tmp_path))
    assert (tmp_path / "dq_results.csv").exists()
    assert (tmp_path / "dim_sample.csv").exists()
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["num_tests"] == 13


def test_determinism_across_runs(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=str(tmp_path / "a"))
    b = run_demo(seed=0, out_dir=str(tmp_path / "b"))
    assert a == b
