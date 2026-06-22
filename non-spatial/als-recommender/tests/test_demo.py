"""Determinism and correctness tests for the synthetic end-to-end demo.

The demo drives the real numpy core (ALS, popularity baseline, ranking metrics)
on seeded synthetic low-rank data, so for a fixed seed every reported number is
reproducible bit-for-bit. These tests pin those committed values and assert the
headline claim the README makes: ALS beats the popularity baseline.
"""

from __future__ import annotations

from pathlib import Path

from recsys.demo import run_demo

# Committed values produced by run_demo(seed=0) on Python 3.14 / numpy 2.4.4 /
# pandas 3.0.3. If the core or the synthesis changes, these update together.
EXPECTED_ALS = {
    "rmse": 0.1917,
    "precision_at_k": 0.0833,
    "recall_at_k": 0.8333,
    "ndcg_at_k": 0.4777,
}
EXPECTED_POP = {
    "rmse": 0.3612,
    "precision_at_k": 0.0,
    "recall_at_k": 0.0,
    "ndcg_at_k": 0.0,
}


def test_run_demo_metrics_are_deterministic(tmp_path: Path) -> None:
    """run_demo(seed=0) reproduces the committed metric values exactly."""
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["k"] == 10
    assert result["als"] == EXPECTED_ALS
    assert result["popularity"] == EXPECTED_POP


def test_als_beats_popularity_on_rmse(tmp_path: Path) -> None:
    """ALS predicts the held-out ratings better (lower RMSE) than popularity."""
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["als"]["rmse"] < result["popularity"]["rmse"]


def test_als_beats_popularity_on_ranking(tmp_path: Path) -> None:
    """ALS clears the non-personalised baseline on every ranking metric."""
    result = run_demo(seed=0, out_dir=tmp_path)
    for metric in ("precision_at_k", "recall_at_k", "ndcg_at_k"):
        assert result["als"][metric] > result["popularity"][metric]


def test_run_demo_writes_artifacts(tmp_path: Path) -> None:
    """The three artefacts are written to the output directory."""
    run_demo(seed=0, out_dir=tmp_path)
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "topn_sample.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
