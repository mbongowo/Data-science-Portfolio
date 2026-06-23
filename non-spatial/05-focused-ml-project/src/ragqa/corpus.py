"""Load the bundled corpus and QA evaluation set (pure stdlib).

The corpus is a folder of short markdown files, one per portfolio project. Each
file's first markdown ``# heading`` (or its filename) is the title; the rest is
the body text. The QA evaluation set is a JSON list of questions paired with the
``doc_id`` (the markdown filename stem) that answers them.
"""

from __future__ import annotations

import json
from pathlib import Path


def _title_and_body(text: str, fallback: str) -> tuple[str, str]:
    """Split a markdown doc into (title, body); first ``# H1`` is the title."""
    lines = text.splitlines()
    title = fallback
    body_lines = lines
    for i, line in enumerate(lines):
        if line.strip().startswith("# "):
            title = line.strip()[2:].strip()
            body_lines = lines[i + 1 :]
            break
    body = "\n".join(body_lines).strip()
    return title, body


def load_corpus(directory: str | Path) -> list[dict]:
    """Read every ``*.md`` file in ``directory`` into document dicts.

    Parameters
    ----------
    directory:
        Folder containing the markdown documents.

    Returns
    -------
    list[dict]
        One dict per file with ``doc_id`` (filename stem), ``title`` and
        ``text`` (the body, heading stripped). Sorted by ``doc_id`` for
        determinism.

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist.
    ValueError
        If it contains no markdown files.
    """
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {path}")

    docs: list[dict] = []
    for md in sorted(path.glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        title, body = _title_and_body(raw, fallback=md.stem)
        docs.append({"doc_id": md.stem, "title": title, "text": body})

    if not docs:
        raise ValueError(f"No markdown documents found in {path}")
    return docs


def load_qa(path: str | Path) -> list[tuple[str, str]]:
    """Read the QA evaluation set into ``(question, relevant_doc_id)`` tuples.

    The JSON file is a list of objects with ``question`` and ``relevant_doc_id``
    keys.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"QA file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return [(item["question"], item["relevant_doc_id"]) for item in data]
