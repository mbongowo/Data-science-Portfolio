"""One-command, fully reproducible demo of the pure-Python clickstream core.

This module generates a small, **seeded synthetic** clickstream and drives the
*real* core functions over it -- :func:`clickstream.windows.tumbling_counts`,
:func:`clickstream.windows.sessionize`, :func:`clickstream.windows.funnel`, and
:func:`clickstream.aggregate.events_per_minute` -- to produce real, reproducible
numbers (not placeholders). It depends only on numpy, pandas, pyyaml, and the
standard library, so it runs anywhere, including CI, with no Kafka or Spark.

The synthetic stream embeds the funnel
``view -> search -> add_to_cart -> checkout -> purchase`` with a realistic
per-step drop-off, so the funnel conversion rates are meaningful. The same
windowing/sessionisation logic is what the Kafka + Spark layer in
:mod:`clickstream.pipeline` applies to a real high-volume stream.

Run it with ``clickstream demo`` or ``run_demo(seed=0)``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from clickstream.aggregate import events_per_minute
from clickstream.windows import funnel, sessionize, tumbling_counts

# Ordered funnel steps (mirrors config/clickstream.yaml). A user must hit these
# in order to advance. Drop-off probabilities below give realistic attrition.
FUNNEL_STEPS: list[str] = ["view", "search", "add_to_cart", "checkout", "purchase"]

# Probability a user who reached step k continues to step k+1.
_ADVANCE_P: list[float] = [0.72, 0.55, 0.60, 0.65]

# Window / session parameters used for the demo (seconds).
TUMBLING_S: float = 60.0
SESSION_GAP_S: float = 1800.0

_N_USERS: int = 200
_BASE_TS: int = 1_700_000_000  # fixed epoch base so timestamps are stable


def _generate_events(seed: int) -> pd.DataFrame:
    """Deterministically generate a small synthetic clickstream DataFrame.

    Each user starts a session at a random offset within a roughly two-hour
    span and walks the funnel, dropping off at each step per ``_ADVANCE_P``.
    Inter-event delays are short (seconds to a couple of minutes). Returns a
    DataFrame with columns ``ts`` (epoch seconds, int), ``user``, ``event``,
    sorted by ``(user, ts)``.
    """
    rng = np.random.default_rng(seed)

    rows: list[tuple[int, str, str]] = []
    for uid in range(_N_USERS):
        user = f"u{uid:03d}"
        # Session start spread across ~2 hours (7200 s).
        t = _BASE_TS + int(rng.integers(0, 7200))
        # Every user views at least once.
        rows.append((t, user, FUNNEL_STEPS[0]))
        depth = 1
        for p in _ADVANCE_P:
            if rng.random() >= p:
                break
            # Advance time by a short, realistic gap (5 s .. 180 s).
            t += int(rng.integers(5, 181))
            rows.append((t, user, FUNNEL_STEPS[depth]))
            depth += 1

        # A few users browse a bit more (extra views/searches) within session.
        for _ in range(int(rng.integers(0, 3))):
            t += int(rng.integers(5, 181))
            extra = FUNNEL_STEPS[int(rng.integers(0, 2))]  # view or search
            rows.append((t, user, extra))

    df = pd.DataFrame(rows, columns=["ts", "user", "event"])
    df = df.sort_values(["user", "ts"], kind="stable").reset_index(drop=True)
    return df


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the end-to-end demo: generate data, drive the real core, write files.

    Parameters
    ----------
    seed:
        Seed for numpy's ``default_rng``; fixes the synthetic stream so the
        returned metrics are deterministic.
    out_dir:
        Directory for the artifacts (created if missing): ``events_per_minute.csv``,
        ``funnel.csv``, and ``summary.json``.

    Returns
    -------
    dict
        Real metrics computed by the core: ``seed``, ``total_events``,
        ``num_users``, ``num_sessions``, ``peak_events_per_min``,
        ``funnel_steps`` (list of ``{step, users, conversion_from_top,
        step_conversion}``), and ``window_s`` / ``session_gap_s``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = _generate_events(seed)
    total_events = int(len(df))
    num_users = int(df["user"].nunique())

    # --- Events per minute (real pandas core) ---------------------------------
    epm = events_per_minute(df)
    peak_events_per_min = int(epm.max())

    # --- Tumbling-window counts (real pure-Python core) -----------------------
    events = list(zip(df["ts"].tolist(), df["event"].tolist(), strict=True))
    tumbling = tumbling_counts(events, TUMBLING_S)
    num_tumbling_windows = int(len(tumbling))

    # --- Sessionisation (real pure-Python core), per user ---------------------
    num_sessions = 0
    for _user, group in df.groupby("user", sort=True):
        ids = sessionize(group["ts"].tolist(), SESSION_GAP_S)
        # session ids start at 0 and increase; count of distinct sessions.
        num_sessions += (max(ids) + 1) if ids else 0
    num_sessions = int(num_sessions)

    # --- Funnel reach (real pure-Python core) ---------------------------------
    user_events: dict[str, list[str]] = defaultdict(list)
    for _user, group in df.groupby("user", sort=True):
        user_events[_user] = group["event"].tolist()
    reach = funnel(user_events, FUNNEL_STEPS)

    top = reach[0] if reach else 0
    funnel_steps = []
    for i, step in enumerate(FUNNEL_STEPS):
        users = int(reach[i])
        conv_top = round(users / top, 4) if top else 0.0
        step_conv = round(users / reach[i - 1], 4) if i > 0 and reach[i - 1] else 1.0
        funnel_steps.append(
            {
                "step": step,
                "users": users,
                "conversion_from_top": conv_top,
                "step_conversion": step_conv,
            }
        )

    metrics = {
        "seed": int(seed),
        "total_events": total_events,
        "num_users": num_users,
        "num_sessions": num_sessions,
        "num_tumbling_windows": num_tumbling_windows,
        "window_s": TUMBLING_S,
        "session_gap_s": SESSION_GAP_S,
        "peak_events_per_min": peak_events_per_min,
        "funnel_steps": funnel_steps,
    }

    # --- Write artifacts ------------------------------------------------------
    epm_df = epm.rename("count").reset_index()
    epm_df.to_csv(out_path / "events_per_minute.csv", index=False)

    pd.DataFrame(funnel_steps).to_csv(out_path / "funnel.csv", index=False)

    with (out_path / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    return metrics


if __name__ == "__main__":  # pragma: no cover - manual entry point
    print(json.dumps(run_demo(0), default=str, indent=2))
