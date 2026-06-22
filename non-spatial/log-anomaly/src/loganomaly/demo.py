"""One-command, end-to-end demo on a small seeded synthetic HDFS-like log.

This drives the REAL numeric core — templating, the event-count matrix, the PCA
reconstruction-error detector, and the precision/recall/F1 metrics — over a tiny
labelled synthetic corpus that is deterministic and runs in well under a second.
It exists so the numbers quoted in the README are reproducible anywhere (incl.
CI) with only numpy / pandas / pyyaml + stdlib: no Spark, no scikit-learn.

The synthetic log imitates the shape of Loghub HDFS_v1: each "session" is one
block's life-cycle made of a handful of recurring templated lines (allocate,
receive, respond, served, deleted) whose variable tokens — block ids, IPs,
sizes — are masked away by :func:`loganomaly.templating.mask_line`. The majority
of sessions are NORMAL (the usual event mix); a minority are ANOMALOUS and
deviate from it (missing the response/served events, or firing a rare
exception/redundant-addition event). The PCA detector, fit only on the
event-count matrix with no labels, then flags the sessions that fall outside the
dominant low-rank "normal" subspace; we score those flags against the recorded
ground-truth labels.

The full Spark pipeline (``loganomaly parse|detect|evaluate``) runs the
*identical* templating + PCA detection on the real labelled Loghub HDFS set; this
demo is the same code path at toy scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from loganomaly.detect import flag, pca_reconstruction_error
from loganomaly.evaluate import confusion_matrix, precision_recall_f1
from loganomaly.features import event_count_matrix
from loganomaly.templating import template_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.random import Generator

# --- Templated line generators -------------------------------------------------
# Each returns a raw log line with variable tokens (block id, ip:port, size) that
# the masker collapses to <*>, so lines of the same kind share one template.

_PORTS = (50010, 50011, 50012)


def _ip(rng: Generator) -> str:
    return ".".join(str(int(x)) for x in rng.integers(1, 255, size=4))


def _allocate(rng: Generator, blk: str) -> str:
    return f"NameSystem.allocateBlock: /user/root/file {rng.integers(0, 9999)} {blk}"


def _receiving(rng: Generator, blk: str) -> str:
    return (
        f"Receiving block {blk} src: /{_ip(rng)}:{rng.choice(_PORTS)} "
        f"dest: /{_ip(rng)}:{rng.choice(_PORTS)} size {rng.integers(1024, 1 << 26)}"
    )


def _received(rng: Generator, blk: str) -> str:
    return (
        f"Received block {blk} of size {rng.integers(1024, 1 << 26)} from /{_ip(rng)}"
    )


def _responder(rng: Generator, blk: str) -> str:
    return f"PacketResponder {rng.integers(0, 3)} for block {blk} terminating"


def _served(rng: Generator, blk: str) -> str:
    return f"{blk} Served block to /{_ip(rng)}:{rng.choice(_PORTS)}"


def _deleted(rng: Generator, blk: str) -> str:
    return (
        f"BLOCK* NameSystem.delete: {blk} is added to invalidSet of "
        f"/{_ip(rng)}:{rng.choice(_PORTS)}"
    )


# Rare events that mark an anomalous block.
def _exception(rng: Generator, blk: str) -> str:
    return (
        f"writeBlock {blk} received exception java.io.IOException: "
        f"Connection reset by peer at /{_ip(rng)}:{rng.choice(_PORTS)}"
    )


def _redundant(rng: Generator, blk: str) -> str:
    return (
        f"Redundant addStoredBlock request received for {blk} on "
        f"/{_ip(rng)}:{rng.choice(_PORTS)} size {rng.integers(1024, 1 << 26)}"
    )


def _normal_session(rng: Generator, blk: str) -> list[str]:
    """A healthy block: allocate, a few receives + a responder, served, deleted."""
    lines = [_allocate(rng, blk)]
    n_replicas = int(rng.integers(2, 4))  # 2-3 receiving datanodes
    for _ in range(n_replicas):
        lines.append(_receiving(rng, blk))
        lines.append(_received(rng, blk))
    lines.append(_responder(rng, blk))
    lines.append(_served(rng, blk))
    if rng.random() < 0.5:
        lines.append(_deleted(rng, blk))
    # A small fraction of healthy blocks legitimately fire a one-off rare event
    # (e.g. a benign redundant addition). These are the false positives that keep
    # precision honestly below 1.0 — rare-but-benign looks like rare-and-bad.
    if rng.random() < 0.06:
        lines.append(_redundant(rng, blk))
    return lines


def _anomalous_session(rng: Generator, blk: str) -> list[str]:
    """A block that deviates: drop the responder/served and fire a rare event."""
    lines = [_allocate(rng, blk)]
    n_replicas = int(rng.integers(1, 3))
    for _ in range(n_replicas):
        lines.append(_receiving(rng, blk))
        # the matching 'Received' acknowledgement is frequently missing
        if rng.random() < 0.4:
            lines.append(_received(rng, blk))
    # Most anomalies fire a rare exception; a subtler minority only *omit* the
    # normal terminating/served events without any rare line, so they sit close
    # to the normal subspace and are sometimes missed (the false negatives).
    if rng.random() < 0.82:
        lines.append(_exception(rng, blk))
        if rng.random() < 0.6:
            lines.append(_redundant(rng, blk))
    else:
        # subtle anomaly: a near-complete-but-truncated block
        lines.append(_responder(rng, blk))
    return lines


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict[str, Any]:
    """Synthesize a labelled log, run the real core, and return the metrics.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng``; fully determines the corpus and
        therefore the metrics.
    out_dir:
        Directory for the written artefacts (created if needed):
        ``templates.csv``, ``scores.csv``, ``summary.json``.

    Returns
    -------
    dict
        ``num_sessions``, ``num_templates``, ``num_anomalies_true``,
        ``precision``, ``recall``, ``f1`` and the confusion-matrix counts
        ``tn, fp, fn, tp``.
    """
    rng = np.random.default_rng(seed)

    n_sessions = 300
    anomaly_rate = 0.12  # ~12% anomalous blocks, HDFS-like minority

    # --- 1. Synthesize the labelled corpus ------------------------------------
    sessions: dict[str, list[str]] = {}
    labels: list[bool] = []
    for _ in range(n_sessions):
        blk = f"blk_{rng.integers(-(1 << 62), 1 << 62)}"
        is_anom = bool(rng.random() < anomaly_rate)
        if is_anom:
            sessions[blk] = _anomalous_session(rng, blk)
        else:
            sessions[blk] = _normal_session(rng, blk)
        labels.append(is_anom)
    y_true = np.array(labels, dtype=bool)

    # --- 2. Drive the REAL core ----------------------------------------------
    # 2a. Templating: mask every line to a template and assign stable ids.
    table: dict[str, int] = {}
    session_to_ids: dict[str, list[int]] = {}
    for blk, lines in sessions.items():
        session_to_ids[blk] = [template_id(line, table) for line in lines]
    n_templates = len(table)

    # 2b. Event-count matrix (sessions x templates).
    X = event_count_matrix(session_to_ids, n_templates)

    # 2c. PCA reconstruction-error detector + quantile flag (no labels used).
    k = 3
    quantile = 0.85
    errors = pca_reconstruction_error(X, k=k)
    y_pred = flag(errors, quantile=quantile)

    # 2d. Score the flags against the ground-truth labels.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    # --- 3. Write artefacts ---------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    templates_df = pd.DataFrame(
        sorted(table.items(), key=lambda kv: kv[1]), columns=["template", "template_id"]
    )
    templates_df.to_csv(out / "templates.csv", index=False)

    scores_df = pd.DataFrame(
        {
            "session": list(session_to_ids.keys()),
            "score": errors,
            "flag": y_pred,
            "label": y_true,
        }
    )
    scores_df.to_csv(out / "scores.csv", index=False)

    result: dict[str, Any] = {
        "num_sessions": int(n_sessions),
        "num_templates": int(n_templates),
        "num_anomalies_true": int(y_true.sum()),
        "detector": f"PCA reconstruction error, k={k}, flag > {quantile:.2f} quantile",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return result
