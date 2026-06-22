r"""Known-answer tests for the lexicon scorer.

The scorer sums token valences (flipping any within a negation window) and
squashes the sum ``s`` with ``s / sqrt(s**2 + alpha)``. With the built-in
DEMO_LEXICON (good=+2, great=+3, bad=-2, terrible=-3) and the default
``alpha = 15`` the expected values below are hand-derived:

* "great"      -> s = +3 -> 3 / sqrt(9 + 15)  = 3 / sqrt(24)  = +0.61237...
* "not great"  -> the negation flips great    -> s = -3 -> -3 / sqrt(24) = -0.61237...
* neutral text -> no lexicon hits             -> s = 0 -> 0.0 exactly
"""

from __future__ import annotations

import math

import pytest

from sentiment.lexicon import DEMO_LEXICON, score_text

# 3 / sqrt(3**2 + 15) = 3 / sqrt(24)
_GREAT = 3.0 / math.sqrt(24.0)


def test_positive_word_is_positive() -> None:
    assert score_text("great") == pytest.approx(_GREAT, abs=1e-12)
    assert score_text("great") > 0.0


def test_negation_flips_sign() -> None:
    # "not great": great falls inside the negation window, so its +3 -> -3.
    assert score_text("not great") == pytest.approx(-_GREAT, abs=1e-12)
    assert score_text("not great") < 0.0


def test_neutral_text_is_zero() -> None:
    # No token is in the lexicon, so the raw sum is 0 and the score is exactly 0.
    assert score_text("the meeting is at noon") == 0.0
    assert score_text("") == 0.0


def test_negation_window_is_bounded() -> None:
    # With negation_window=3, "good" is the 4th token after "not" and is NOT
    # flipped: tokens are [not, a, b, c, good]; "good" sits outside the window.
    assert score_text("not a b c good") == pytest.approx(
        2.0 / math.sqrt(4.0 + 15.0), abs=1e-12
    )


def test_valences_accumulate() -> None:
    # "great" (+3) and "terrible" (-3) cancel to s = 0.
    assert score_text("great terrible") == 0.0
    # "good" (+2) and "great" (+3) sum to s = +5.
    assert score_text("good great") == pytest.approx(
        5.0 / math.sqrt(25.0 + 15.0), abs=1e-12
    )


def test_custom_lexicon_is_used() -> None:
    custom = {"meh": -1.0}
    assert score_text("meh", custom) == pytest.approx(
        -1.0 / math.sqrt(1.0 + 15.0), abs=1e-12
    )
    # DEMO_LEXICON words are not in the custom lexicon, so they score 0.
    assert score_text("great", custom) == 0.0
    assert "great" in DEMO_LEXICON
