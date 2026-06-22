"""Text normalisation and tokenisation.

Pure-stdlib helpers that turn a raw comment string into a clean token list. The
same normalisation is applied before lexicon scoring, before model inference,
and before TF-IDF, so that the three stages agree on what a "token" is.

The normalisation is deliberately conservative and deterministic:

1. lowercase;
2. strip URLs (``http(s)://...`` and bare ``www....``);
3. strip Markdown control characters Reddit uses (``* _ ` ~ # > []()``) while
   keeping the visible text;
4. drop the remaining punctuation, replacing it with a space so words do not get
   glued together;
5. collapse any run of whitespace to a single space and trim.

These steps are covered by known-answer tests; the exact output for a given
input is part of the contract.
"""

from __future__ import annotations

import re

# Order matters: URLs are stripped before punctuation, or the "//" and "."
# inside a URL would survive as fragments.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
# Markdown link text in [text](url): keep the visible text, drop the target.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Anything that is not a word character or whitespace becomes a space.
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Normalise a raw string to lowercase, punctuation-free, single-spaced text.

    Parameters
    ----------
    s:
        Raw document text (a Reddit comment body, a tweet, ...).

    Returns
    -------
    str
        The normalised text. Empty input (or input that is entirely URLs and
        punctuation) returns the empty string.

    Examples
    --------
    >>> normalize_text("Check https://x.com -- it's GREAT!!!")
    'check its great'
    >>> normalize_text("[NASA](http://nasa.gov) said *no*")
    'nasa said no'
    """
    if not s:
        return ""
    text = s.lower()
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(s: str) -> list[str]:
    """Normalise ``s`` and split it into whitespace-delimited tokens.

    Parameters
    ----------
    s:
        Raw document text.

    Returns
    -------
    list[str]
        The list of tokens after :func:`normalize_text`. Empty input returns an
        empty list (not ``[""]``).

    Examples
    --------
    >>> tokenize("Not great, really.")
    ['not', 'great', 'really']
    >>> tokenize("   ")
    []
    """
    normalized = normalize_text(s)
    if not normalized:
        return []
    return normalized.split(" ")
