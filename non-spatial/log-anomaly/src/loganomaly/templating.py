"""Drain-lite log templating: collapse raw lines to stable event templates.

Real log mining (Drain, Spell, IPLoM) clusters lines into templates by a parse
tree. This module is the small, dependency-free core of that idea: mask the
*variable* tokens in a line with ``"<*>"`` so that lines that differ only in
their values collapse to one template, then map each distinct masked template to
a stable integer id.

The masking is regex-based and deliberately conservative; it targets the token
classes that vary most in infrastructure logs:

* block ids (``blk_-?\\d+``) — HDFS-style,
* hex ids (``0x...`` and long hex runs),
* IPv4 addresses and ``IPv4:port`` pairs,
* bare numbers (integers and floats).

It returns the masked string only; it does no inference and has no third-party
dependency, which is why the known-answer tests can exercise it directly.
"""

from __future__ import annotations

import re

# Order matters: more specific patterns (block ids, ip:port) run before the
# generic number pass so they are not chewed up by it.
_PLACEHOLDER = "<*>"

# blk_123 or blk_-9214 (HDFS block ids carry an optional leading minus).
_RE_BLOCK_ID = re.compile(r"\bblk_-?\d+\b")
# 0xdeadbeef and long bare hex runs (>= 6 hex digits, not a plain decimal).
_RE_HEX_ID = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{6,}\b")
# IPv4 with an optional :port, optionally preceded by a leading slash.
_RE_IP_PORT = re.compile(r"/?\b\d{1,3}(?:\.\d{1,3}){3}\b(?::\d+)?")
# Bare integers and floats (run last).
_RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def mask_line(line: str) -> str:
    """Replace variable tokens in ``line`` with ``"<*>"`` and return the template.

    The passes run in a fixed order so that specific token classes are masked
    before the generic number pass:

    1. block ids (``blk_-?\\d+``),
    2. hex ids (``0x...`` or a long hex run),
    3. IPv4 / ``IPv4:port`` (an optional leading ``/`` is consumed too),
    4. bare numbers.

    Consecutive whitespace is collapsed so that templates compare cleanly.

    Examples
    --------
    >>> mask_line("Receiving block blk_123 src: /10.0.0.1:50010 size 4096")
    'Receiving block <*> src: <*> size <*>'

    Parameters
    ----------
    line:
        A single raw log line (any trailing newline is stripped).

    Returns
    -------
    str
        The masked template string.
    """
    s = line.rstrip("\n")
    s = _RE_BLOCK_ID.sub(_PLACEHOLDER, s)
    s = _RE_HEX_ID.sub(_PLACEHOLDER, s)
    s = _RE_IP_PORT.sub(_PLACEHOLDER, s)
    s = _RE_NUMBER.sub(_PLACEHOLDER, s)
    # Collapse runs of whitespace introduced or left by masking.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def template_id(line: str, table: dict[str, int]) -> int:
    """Return a stable integer id for the template of ``line``.

    The line is masked with :func:`mask_line`, then looked up in ``table`` (a
    mutable ``template -> id`` dictionary). A template seen before returns its
    existing id; a new template is assigned the next id (``len(table)``) and
    inserted. ``table`` is mutated in place, so the same dictionary threaded
    across a corpus yields a consistent id space.

    Parameters
    ----------
    line:
        A single raw log line.
    table:
        Mutable mapping from masked template to integer id. Pass the same dict
        across calls to keep ids stable.

    Returns
    -------
    int
        The id for this line's template.
    """
    template = mask_line(line)
    if template in table:
        return table[template]
    new_id = len(table)
    table[template] = new_id
    return new_id
