"""Deterministic test for the seeded synthetic demo.

The demo drives the *real* pure-Python core over a seeded synthetic clickstream,
so its metrics are fully reproducible. Pinning the exact returned values here
guarantees that the demo runs in CI and that the numbers quoted in the README
stay true. Only numpy/pandas + stdlib are required (no Kafka/Spark).
"""

from __future__ import annotations

import json

from clickstream.demo import run_demo


def test_run_demo_is_deterministic(tmp_path) -> None:
    """run_demo(seed=0) returns the exact committed metrics (also in README)."""
    metrics = run_demo(seed=0, out_dir=tmp_path)

    assert metrics["seed"] == 0
    assert metrics["total_events"] == 659
    assert metrics["num_users"] == 200
    assert metrics["num_sessions"] == 200
    assert metrics["peak_events_per_min"] == 13

    steps = metrics["funnel_steps"]
    reach = {s["step"]: s["users"] for s in steps}
    assert reach == {
        "view": 200,
        "search": 164,
        "add_to_cart": 62,
        "checkout": 36,
        "purchase": 21,
    }
    # Funnel reach is monotonically non-increasing by construction.
    users = [s["users"] for s in steps]
    assert all(users[i] >= users[i + 1] for i in range(len(users) - 1))


def test_run_demo_writes_artifacts(tmp_path) -> None:
    """The demo writes the three expected artifacts into out_dir."""
    run_demo(seed=0, out_dir=tmp_path)

    assert (tmp_path / "events_per_minute.csv").is_file()
    assert (tmp_path / "funnel.csv").is_file()
    summary = tmp_path / "summary.json"
    assert summary.is_file()

    loaded = json.loads(summary.read_text(encoding="utf-8"))
    assert loaded["total_events"] == 659


def test_run_demo_stable_across_calls(tmp_path) -> None:
    """Two seeded runs produce identical metrics (no hidden state)."""
    a = run_demo(seed=0, out_dir=tmp_path / "a")
    b = run_demo(seed=0, out_dir=tmp_path / "b")
    assert a == b
