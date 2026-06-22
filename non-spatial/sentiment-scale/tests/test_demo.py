"""Tests for the seeded synthetic end-to-end demo.

The demo drives the real pure-numpy/pandas core on a deterministic synthetic
corpus, so its key outputs are fixed for ``seed=0`` and committed here. The
posts are assembled from known-valence words, so the lexicon scorer recovers the
planted labels exactly (validation accuracy 1.0), and the planted sentiment
trend shift is recoverable as a positive -> negative flip across ``shift_date``.
"""

from __future__ import annotations

import json

from sentiment.demo import run_demo


def test_run_demo_metrics_are_committed_values(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=tmp_path)

    assert result["num_posts"] == 336
    assert result["validation_accuracy"] == 1.0
    assert result["mean_sentiment_before"] == 0.264167
    assert result["mean_sentiment_after"] == -0.315198
    assert result["shift_date"] == "2019-03-29"
    assert isinstance(result["top_terms"], list) and result["top_terms"]


def test_planted_shift_is_detected(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=tmp_path)
    # Pre-shift corpus leans positive, post-shift leans negative: the mean must
    # cross zero in the planted direction.
    assert result["mean_sentiment_before"] > 0
    assert result["mean_sentiment_after"] < 0
    assert result["mean_sentiment_after"] < result["mean_sentiment_before"]


def test_artifacts_are_written(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)

    ts = tmp_path / "sentiment_timeseries.csv"
    summary = tmp_path / "summary.json"
    assert ts.exists()
    assert summary.exists()

    header = ts.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ["period", "mean_score", "n"]

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["validation_accuracy"] == 1.0


def test_run_demo_is_deterministic(tmp_path) -> None:
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
