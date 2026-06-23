"""Deterministic test for the bundled retrieval demo.

The demo drives the real retrieval core over the bundled portfolio corpus, so
its metrics are fully reproducible. Pinning the key values keeps the README
figures honest and guarantees the demo runs in CI with only numpy + stdlib.
"""

from __future__ import annotations

import json

from ragqa.demo import run_demo


def test_run_demo_pinned_metrics(tmp_path):
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["n_docs"] == 14
    assert result["n_chunks"] == 29
    assert result["vocab_size"] == 648
    assert result["recall_at_3"] == 1.0
    assert result["mrr"] == 1.0


def test_run_demo_quality_bounds(tmp_path):
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["recall_at_3"] >= 0.6
    assert result["mrr"] >= 0.6


def test_run_demo_sample_question_retrieves_expected_doc(tmp_path):
    result = run_demo(seed=0, out_dir=tmp_path)
    # The first sample asks about Sentinel-2 NDVI anomalies -> eo-monitor.
    assert result["sample_top_doc"] == "eo-monitor"


def test_run_demo_writes_artifacts(tmp_path):
    run_demo(seed=0, out_dir=tmp_path)
    assert (tmp_path / "eval.json").is_file()
    answers = tmp_path / "sample_answers.json"
    assert answers.is_file()
    loaded = json.loads(answers.read_text(encoding="utf-8"))
    assert loaded[0]["top_doc"] == "eo-monitor"


def test_run_demo_stable_across_calls(tmp_path):
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
