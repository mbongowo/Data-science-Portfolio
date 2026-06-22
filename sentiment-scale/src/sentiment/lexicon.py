"""Lexicon-based sentiment scoring (a small, transparent VADER-style scorer).

This is the pure-numpy/stdlib reference scorer. It sums the valences of matched
tokens, flips the sign of tokens that fall inside a negation window, and squashes
the raw sum into roughly ``[-1, 1]`` with the VADER normalisation

.. math::

    \\text{score} = \\frac{s}{\\sqrt{s^2 + \\alpha}}

where ``s`` is the (negation-adjusted) sum of token valences and ``alpha`` is a
constant (15 by default, as in VADER). The squashing is monotone and odd, so the
sign of the score is the sign of ``s`` and a longer agreeing document saturates
towards +-1 without ever leaving it.

A tiny built-in :data:`DEMO_LEXICON` is provided so the behaviour can be checked
with hand-derived known answers; production runs pass a full lexicon (e.g. the
VADER lexicon) in the same ``{token: valence}`` form.
"""

from __future__ import annotations

import math

from sentiment.clean import tokenize

#: Minimal demo lexicon for tests and examples. Real runs supply a full one.
DEMO_LEXICON: dict[str, float] = {
    "good": 2.0,
    "great": 3.0,
    "bad": -2.0,
    "terrible": -3.0,
}

#: Words that negate (flip the valence of) the tokens that follow them.
NEGATIONS: frozenset[str] = frozenset({"not", "no", "never"})


def score_text(
    text: str,
    lexicon: dict[str, float] | None = None,
    *,
    negation_window: int = 3,
    alpha: float = 15.0,
) -> float:
    """Score ``text`` in roughly ``[-1, 1]`` using a valence lexicon.

    The text is tokenised with :func:`sentiment.clean.tokenize`. Each token that
    appears in ``lexicon`` contributes its valence. If a negation word
    (``not``/``no``/``never``) appears, the valence of every lexicon token within
    the next ``negation_window`` tokens is flipped. The negation-adjusted sum is
    then normalised by ``s / sqrt(s**2 + alpha)``.

    Parameters
    ----------
    text:
        Raw document text.
    lexicon:
        Mapping ``{token: valence}``. Defaults to :data:`DEMO_LEXICON`.
    negation_window:
        Number of tokens after a negation word whose valence is flipped.
    alpha:
        VADER-style normalisation constant; larger ``alpha`` squashes harder.

    Returns
    -------
    float
        The normalised sentiment score. ``0.0`` for text with no lexicon hits
        (or an empty document); positive for net-positive valence, negative for
        net-negative.

    Examples
    --------
    >>> round(score_text("great"), 4)
    0.6124
    >>> round(score_text("not great"), 4)
    -0.6124
    >>> score_text("the meeting is at noon")
    0.0
    """
    if lexicon is None:
        lexicon = DEMO_LEXICON

    tokens = tokenize(text)
    if not tokens:
        return 0.0

    total = 0.0
    # Number of remaining tokens (counting the current one) whose valence is
    # flipped by a preceding negation word.
    flip_remaining = 0
    for token in tokens:
        if token in NEGATIONS:
            # The negation applies to the *following* window tokens.
            flip_remaining = negation_window
            continue
        if token in lexicon:
            valence = lexicon[token]
            if flip_remaining > 0:
                valence = -valence
            total += valence
        if flip_remaining > 0:
            flip_remaining -= 1

    if total == 0.0:
        return 0.0
    return total / math.sqrt(total * total + alpha)
