"""Reproducibility tests for the one-command demo.

``run_demo(seed=0)`` drives the real pure-numpy core on a seeded stochastic
block model with three planted communities. The metrics are deterministic, so we
pin them to the values committed in the README and assert that label propagation
recovers exactly the planted number of communities. numpy / pandas / stdlib only.
"""

from __future__ import annotations

import json

import pytest

from bdgraph.demo import run_demo, synthesize_sbm

# Committed reference values for seed 0 (also reported in README "Result first").
EXPECTED = {
    "num_nodes": 30,
    "num_edges": 83,
    "num_components": 1,
    "num_communities_found": 3,
    "num_planted_communities": 3,
    "global_triangles": 69,
    "max_core_number": 5,
    "top_betweenness_node": 16,
    "max_degree": 9,
}


def test_run_demo_metrics_match_committed_values(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=tmp_path)
    for key, value in EXPECTED.items():
        assert result[key] == value, f"{key}: {result[key]!r} != {value!r}"


def test_label_propagation_recovers_planted_communities(tmp_path) -> None:
    """Detected community count equals the planted count (clean SBM)."""
    result = run_demo(seed=0, out_dir=tmp_path)
    assert result["num_communities_found"] == result["num_planted_communities"]


def test_structural_metrics_are_committed(tmp_path) -> None:
    """The new structural diagnostics match their committed seed-0 values."""
    result = run_demo(seed=0, out_dir=tmp_path)
    # On a clean SBM the detected partition equals the planted one, so their
    # modularity is identical and high-positive.
    assert result["modularity_found"] == result["modularity_planted"]
    assert result["modularity_found"] == pytest.approx(0.559225, abs=1e-6)
    assert result["mean_degree"] == pytest.approx(5.533333, abs=1e-6)
    # The top-betweenness node coincides with the top-PageRank hub of the densest
    # block.
    assert result["top_betweenness_node"] == result["top_pagerank"][0][0]


def test_top_pagerank_is_sorted_descending(tmp_path) -> None:
    result = run_demo(seed=0, out_dir=tmp_path)
    top = result["top_pagerank"]
    assert len(top) == 5
    scores = [score for _node, score in top]
    assert scores == sorted(scores, reverse=True)
    assert top[0][0] == 16  # committed top node for seed 0


def test_artifacts_written(tmp_path) -> None:
    run_demo(seed=0, out_dir=tmp_path)
    for name in ("pagerank_top.csv", "communities.csv", "summary.json"):
        assert (tmp_path / name).exists(), f"missing artifact {name}"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_nodes"] == EXPECTED["num_nodes"]


def test_deterministic_across_runs(tmp_path) -> None:
    first = run_demo(seed=0, out_dir=tmp_path / "a")
    second = run_demo(seed=0, out_dir=tmp_path / "b")
    assert first["top_pagerank"] == second["top_pagerank"]
    assert first["global_triangles"] == second["global_triangles"]


def test_sbm_is_symmetric_simple_graph() -> None:
    a = synthesize_sbm(seed=0)
    assert (a == a.T).all()  # undirected
    assert (a.diagonal() == 0).all()  # no self-loops
    assert ((a == 0) | (a == 1)).all()  # simple (0/1)
