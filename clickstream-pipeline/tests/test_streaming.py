"""Known-answer tests for the bounded-memory / event-time streaming primitives.

Every expected value below is *hand-derived* on a tiny input and matches the
worked example written into the corresponding function docstring, so a green test
proves the logic rather than merely that it runs. No streaming engine (pyspark,
kafka) is required.

Worked examples mirrored here:

1. top_k_heavy_hitters([a,a,a,b,b,c,a,c,c], k=2, counters=2) -> [(a,4),(c,3)]
   (Misra-Gries with 2 counters keeps {a, c}; exact pass ranks a=4, c=3.)

2. reorder_within_lateness, allowed_lateness=2, arrivals
   (10,a),(12,b),(11,c),(9,d),(13,e). Watermark = max_ts - 2; (9,d) arrives when
   the watermark is 10 so 9 < 10 is dropped. Drain emits
   [(10,a),(11,c),(12,b),(13,e)] with dropped=1.

3. funnel_time_to_convert, steps [view,cart,buy]:
   view->cart medians over {10,20,0} = 10; cart->buy over {20,45} = 32.5.
   -> [10.0, 32.5].

4. retention, period_s=100:
   P0={u1,u2,u3}, P1={u1,u2}, P2={u2,u4}.
   0->1 = 2/3, 1->2 = 1/2 -> [0.6667, 0.5].
"""

from __future__ import annotations

import math

import pytest

from clickstream import (
    funnel_time_to_convert,
    reorder_within_lateness,
    retention,
    top_k_heavy_hitters,
)

# --------------------------------------------------------------------------- #
# top_k_heavy_hitters                                                         #
# --------------------------------------------------------------------------- #


def test_top_k_heavy_hitters_known() -> None:
    """Hand-derived Misra-Gries (2 counters) + exact pass -> [(a, 4), (c, 3)]."""
    keys = ["a", "a", "a", "b", "b", "c", "a", "c", "c"]
    assert top_k_heavy_hitters(keys, 2, counters=2) == [("a", 4), ("c", 3)]


def test_top_k_heavy_hitters_default_capacity_is_exact_for_skew() -> None:
    """The generous default capacity returns the exact top-k on a skewed stream."""
    keys = ["a", "a", "a", "b", "b", "c", "a", "c", "c"]  # a=4, c=3, b=2
    assert top_k_heavy_hitters(keys, 2) == [("a", 4), ("c", 3)]


def test_top_k_heavy_hitters_finds_global_mode_k1() -> None:
    """k=1 returns the exact single most-frequent key under the default capacity."""
    keys = ["x", "y", "x", "z", "x", "y"]  # x=3, y=2, z=1
    assert top_k_heavy_hitters(keys, 1) == [("x", 3)]


def test_top_k_heavy_hitters_threshold_guarantee_small_summary() -> None:
    """A heavy hitter above n/(counters+1) survives a deliberately tiny summary.

    n = 12, counters = 1 -> threshold n/2 = 6. 'a' occurs 7 times (> 6), so it is
    guaranteed to survive and be reported exactly, even with a single counter.
    """
    keys = ["a"] * 7 + ["b", "c", "d", "e", "f"]
    assert top_k_heavy_hitters(keys, 1, counters=1) == [("a", 7)]


def test_top_k_heavy_hitters_counts_are_exact() -> None:
    """Returned counts are the true frequencies, not Misra-Gries lower bounds."""
    keys = ["a"] * 10 + ["b"] * 6 + ["c"] * 3 + ["d"] * 1
    result = top_k_heavy_hitters(keys, 3)
    assert result == [("a", 10), ("b", 6), ("c", 3)]


def test_top_k_heavy_hitters_ties_break_by_key() -> None:
    """Equal counts are ordered by ascending key for determinism."""
    keys = ["b", "a", "b", "a", "c"]  # a=2, b=2, c=1
    assert top_k_heavy_hitters(keys, 2) == [("a", 2), ("b", 2)]


def test_top_k_heavy_hitters_k_exceeds_distinct() -> None:
    """Asking for more than the number of distinct keys returns them all sorted."""
    result = top_k_heavy_hitters(["a", "a", "b"], 10)
    assert result == [("a", 2), ("b", 1)]


def test_top_k_heavy_hitters_empty() -> None:
    """Empty stream -> empty result."""
    assert top_k_heavy_hitters([], 3) == []


def test_top_k_heavy_hitters_single_element() -> None:
    assert top_k_heavy_hitters(["a"], 3) == [("a", 1)]


def test_top_k_heavy_hitters_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        top_k_heavy_hitters(["a"], 0)
    with pytest.raises(ValueError):
        top_k_heavy_hitters(["a"], -1)


def test_top_k_heavy_hitters_rejects_bad_counters() -> None:
    with pytest.raises(ValueError):
        top_k_heavy_hitters(["a"], 1, counters=0)


# --------------------------------------------------------------------------- #
# reorder_within_lateness                                                     #
# --------------------------------------------------------------------------- #


def test_reorder_within_lateness_known() -> None:
    """Hand-derived: one late drop, rest emitted in ts order."""
    events = [(10.0, "a"), (12.0, "b"), (11.0, "c"), (9.0, "d"), (13.0, "e")]
    ordered, dropped = reorder_within_lateness(events, 2.0)
    assert ordered == [(10.0, "a"), (11.0, "c"), (12.0, "b"), (13.0, "e")]
    assert dropped == 1


def test_reorder_already_sorted_no_drops() -> None:
    """An in-order stream is returned unchanged with zero drops."""
    events = [(0.0, "a"), (1.0, "b"), (2.0, "c")]
    ordered, dropped = reorder_within_lateness(events, 5.0)
    assert ordered == events
    assert dropped == 0


def test_reorder_stable_on_equal_timestamps() -> None:
    """Equal-ts events keep arrival order (stable)."""
    events = [(5.0, "first"), (5.0, "second"), (5.0, "third")]
    ordered, dropped = reorder_within_lateness(events, 0.0)
    assert ordered == events
    assert dropped == 0


def test_reorder_zero_lateness_drops_any_older_than_max() -> None:
    """With allowed_lateness=0, anything strictly older than the running max drops."""
    events = [(10.0, "a"), (8.0, "b"), (10.0, "c")]
    ordered, dropped = reorder_within_lateness(events, 0.0)
    # 8 < watermark(=10) once max is 10 -> dropped.
    assert dropped == 1
    assert ordered == [(10.0, "a"), (10.0, "c")]


def test_reorder_all_late_after_a_big_jump() -> None:
    """A large early timestamp pushes the watermark so later small ones all drop."""
    events = [(100.0, "big"), (1.0, "x"), (2.0, "y"), (3.0, "z")]
    ordered, dropped = reorder_within_lateness(events, 1.0)
    assert ordered == [(100.0, "big")]
    assert dropped == 3


def test_reorder_empty() -> None:
    assert reorder_within_lateness([], 5.0) == ([], 0)


def test_reorder_single_element() -> None:
    assert reorder_within_lateness([(7.0, "a")], 5.0) == ([(7.0, "a")], 0)


def test_reorder_rejects_negative_lateness() -> None:
    with pytest.raises(ValueError):
        reorder_within_lateness([(0.0, "a")], -1.0)


# --------------------------------------------------------------------------- #
# funnel_time_to_convert                                                      #
# --------------------------------------------------------------------------- #


def test_funnel_time_to_convert_known() -> None:
    """Hand-derived medians -> [10.0, 32.5]."""
    user_events = {
        "u1": [(0.0, "view"), (10.0, "cart"), (30.0, "buy")],
        "u2": [(0.0, "view"), (20.0, "cart")],
        "u3": [(5.0, "view"), (5.0, "cart"), (50.0, "buy")],
    }
    steps = ["view", "cart", "buy"]
    assert funnel_time_to_convert(user_events, steps) == [10.0, 32.5]


def test_funnel_time_to_convert_odd_count_median() -> None:
    """Three durations -> the middle one."""
    user_events = {
        "u1": [(0.0, "view"), (10.0, "cart")],
        "u2": [(0.0, "view"), (20.0, "cart")],
        "u3": [(0.0, "view"), (30.0, "cart")],
    }
    assert funnel_time_to_convert(user_events, ["view", "cart"]) == [20.0]


def test_funnel_time_to_convert_uncompleted_transition_is_none() -> None:
    """A transition no user completes yields None."""
    user_events = {"u1": [(0.0, "view"), (5.0, "cart")]}  # nobody buys
    result = funnel_time_to_convert(user_events, ["view", "cart", "buy"])
    assert result == [5.0, None]


def test_funnel_time_to_convert_requires_in_order() -> None:
    """Steps occurring out of order do not count as a completed transition."""
    # buy precedes cart, so cart->buy is never completed; view->cart still is.
    user_events = {"u1": [(0.0, "view"), (10.0, "buy"), (20.0, "cart")]}
    result = funnel_time_to_convert(user_events, ["view", "cart", "buy"])
    assert result == [20.0, None]


def test_funnel_time_to_convert_empty_users() -> None:
    """No users -> a None per transition."""
    assert funnel_time_to_convert({}, ["view", "cart", "buy"]) == [None, None]


def test_funnel_time_to_convert_rejects_short_steps() -> None:
    with pytest.raises(ValueError):
        funnel_time_to_convert({"u1": [(0.0, "view")]}, ["view"])


# --------------------------------------------------------------------------- #
# retention                                                                   #
# --------------------------------------------------------------------------- #


def test_retention_known() -> None:
    """Hand-derived retention -> [2/3, 1/2]."""
    users = {
        "u1": [10.0, 120.0],
        "u2": [50.0, 150.0, 250.0],
        "u3": [30.0],
        "u4": [220.0],
    }
    result = retention(users, 100.0)
    assert len(result) == 2
    assert math.isclose(result[0], 2 / 3)
    assert math.isclose(result[1], 1 / 2)


def test_retention_full_then_none() -> None:
    """All users return once, then none -> [1.0, 0.0]."""
    users = {
        "u1": [0.0, 100.0],
        "u2": [10.0, 110.0],
    }
    # Periods: P0={u1,u2}, P1={u1,u2}. Only one transition, full retention.
    assert retention(users, 100.0) == [1.0]


def test_retention_empty_intermediate_period_is_zero() -> None:
    """A gap period with no activity gives 0.0 retention from the prior period."""
    users = {"u1": [0.0, 250.0]}  # active in P0 and P2, not P1
    # span lo=0, hi=2: transitions 0->1 and 1->2.
    # P0={u1}, P1={}, P2={u1}. 0->1 = 0/1 = 0; 1->2: source P1 empty -> 0.0.
    assert retention(users, 100.0) == [0.0, 0.0]


def test_retention_single_period_returns_empty() -> None:
    """All activity in one period -> no transitions -> empty list."""
    users = {"u1": [0.0, 50.0], "u2": [10.0]}
    assert retention(users, 100.0) == []


def test_retention_no_activity_returns_empty() -> None:
    assert retention({}, 100.0) == []
    assert retention({"u1": []}, 100.0) == []


def test_retention_rejects_nonpositive_period() -> None:
    with pytest.raises(ValueError):
        retention({"u1": [0.0]}, 0.0)
