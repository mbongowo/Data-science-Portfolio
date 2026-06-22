"""Known-answer tests for text normalisation and tokenisation.

The expected output for each input is written out by hand: normalisation is
deterministic, so a green test pins the exact contract (lowercase, URLs and
Markdown stripped, punctuation dropped, whitespace collapsed) rather than just
checking that the function runs.
"""

from __future__ import annotations

from sentiment.clean import normalize_text, tokenize


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize_text("  Hello   WORLD  ") == "hello world"


def test_normalize_strips_urls() -> None:
    # The URL is removed entirely; surrounding words survive.
    assert normalize_text("see https://example.com/x now") == "see now"
    assert normalize_text("visit www.example.com please") == "visit please"


def test_normalize_keeps_markdown_link_text_drops_target() -> None:
    assert normalize_text("[NASA](http://nasa.gov) said no") == "nasa said no"


def test_normalize_drops_punctuation_without_gluing_words() -> None:
    # Punctuation becomes a space, so "it's GREAT!!!" -> "it s great".
    assert normalize_text("Check -- it's GREAT!!!") == "check it s great"


def test_normalize_empty_and_punctuation_only() -> None:
    assert normalize_text("") == ""
    assert normalize_text("!!!???") == ""


def test_tokenize_splits_on_whitespace() -> None:
    assert tokenize("Not great, really.") == ["not", "great", "really"]


def test_tokenize_empty_returns_empty_list() -> None:
    # Must be [] not [""]; downstream code relies on len() == 0 for empty docs.
    assert tokenize("   ") == []
    assert tokenize("https://only.a.url") == []
