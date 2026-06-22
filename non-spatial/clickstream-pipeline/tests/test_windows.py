"""Known-answer tests for the pure-Python windowing and funnel primitives.

These tests use *hand-derived* expected values on tiny inputs, so a green test
proves the logic, not merely that it runs. They have **no third-party engine
dependency** (no pyspark, no kafka), so they always execute.

Worked examples used below:

1. Tumbling counts. Events at ts = [0, 5, 10, 12, 25] with window_s = 10.
   Window start = floor(ts/10)*10:
     0 -> 0, 5 -> 0, 10 -> 10, 12 -> 10, 25 -> 20.
   => {0: 2, 10: 2, 20: 1}.

2. Sliding counts. Events at ts = [0, 7] with window_s = 10, slide_s = 5.
   Window starts are multiples of 5. A window [s, s+10) contains ts.
     ts=0: starts in (-10, 0] that are multiples of 5 and <= 0: {-5, 0}.
     ts=7: starts in (-3, 7] multiples of 5 and <= 7: {0, 5}.
   Tally: -5 -> 1 (from 0), 0 -> 2 (from 0 and 7), 5 -> 1 (from 7).
   => {-5: 1, 0: 2, 5: 1}.

3. Sessionize. [0, 1, 2, 10, 11] with gap_s = 5.
   gaps: 1,1,8,1. Only 8 > 5 -> new session at index 3.
   => [0, 0, 0, 1, 1].

4. Funnel. steps = [view, cart, buy].
   users:
     a: [view, cart, buy]      -> depth 3 (reaches all 3)
     b: [view, view, cart]     -> depth 2 (view then cart)
     c: [view]                 -> depth 1
     d: [cart, buy]            -> depth 0 (never sees `view` first)
   reach prefix-1 = {a,b,c} = 3; prefix-2 = {a,b} = 2; prefix-3 = {a} = 1.
   => [3, 2, 1].
"""

from __future__ import annotations

import pytest

from clickstream import funnel, sessionize, sliding_counts, tumbling_counts


def test_tumbling_counts_known() -> None:
    """Hand-derived tumbling counts: {0: 2, 10: 2, 20: 1}."""
    events = [(0.0, "a"), (5.0, "a"), (10.0, "b"), (12.0, "b"), (25.0, "c")]
    assert tumbling_counts(events, 10.0) == {0.0: 2, 10.0: 2, 20.0: 1}


def test_tumbling_boundary_goes_to_later_window() -> None:
    """An event exactly on a boundary lands in the later window (half-open)."""
    assert tumbling_counts([(10.0, "x")], 10.0) == {10.0: 1}


def test_tumbling_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        tumbling_counts([(0.0, "x")], 0.0)


def test_tumbling_empty_is_empty() -> None:
    """No events -> no windows."""
    assert tumbling_counts([], 10.0) == {}


def test_tumbling_single_event() -> None:
    assert tumbling_counts([(7.0, "x")], 10.0) == {0.0: 1}


def test_sliding_counts_known() -> None:
    """Hand-derived sliding counts: {-5: 1, 0: 2, 5: 1}."""
    events = [(0.0, "a"), (7.0, "b")]
    assert sliding_counts(events, 10.0, 5.0) == {-5.0: 1, 0.0: 2, 5.0: 1}


def test_sliding_equals_tumbling_when_slide_equals_window() -> None:
    """With slide_s == window_s the sliding result matches tumbling."""
    events = [(0.0, "a"), (5.0, "a"), (10.0, "b"), (12.0, "b"), (25.0, "c")]
    assert sliding_counts(events, 10.0, 10.0) == tumbling_counts(events, 10.0)


def test_sliding_rejects_slide_larger_than_window() -> None:
    with pytest.raises(ValueError):
        sliding_counts([(0.0, "x")], 5.0, 10.0)


def test_sessionize_known() -> None:
    """[0,1,2,10,11], gap_s=5 -> [0,0,0,1,1] (hand-derived)."""
    assert sessionize([0, 1, 2, 10, 11], 5) == [0, 0, 0, 1, 1]


def test_sessionize_gap_exactly_threshold_does_not_split() -> None:
    """A gap of exactly gap_s stays in the same session (strict >)."""
    assert sessionize([0, 5, 10], 5) == [0, 0, 0]


def test_sessionize_rejects_unsorted() -> None:
    with pytest.raises(ValueError):
        sessionize([0, 10, 5], 5)


def test_sessionize_empty_is_empty() -> None:
    assert sessionize([], 5) == []


def test_sessionize_single_event() -> None:
    assert sessionize([42], 5) == [0]


def test_funnel_known() -> None:
    """Hand-derived funnel reach counts -> [3, 2, 1]."""
    user_events = {
        "a": ["view", "cart", "buy"],
        "b": ["view", "view", "cart"],
        "c": ["view"],
        "d": ["cart", "buy"],
    }
    steps = ["view", "cart", "buy"]
    assert funnel(user_events, steps) == [3, 2, 1]


def test_funnel_is_monotone_non_increasing() -> None:
    """Reaching step k implies reaching step k-1, so counts never rise."""
    user_events = {
        "a": ["view", "cart", "buy"],
        "b": ["view", "cart"],
        "c": ["view"],
    }
    counts = funnel(user_events, ["view", "cart", "buy"])
    assert counts == [3, 2, 1]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


def test_funnel_rejects_empty_steps() -> None:
    with pytest.raises(ValueError):
        funnel({"a": ["view"]}, [])


def test_funnel_no_users_is_all_zero() -> None:
    """An empty user map reaches nothing."""
    assert funnel({}, ["view", "cart"]) == [0, 0]
