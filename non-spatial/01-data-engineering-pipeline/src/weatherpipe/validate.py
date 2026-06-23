"""Validation: range / null / duplicate checks on the tidy weather frame.

:func:`validate_weather` mirrors the kind of data-quality gate a dbt project runs
(``not_null``, ``accepted_range``, ``unique``, plus a cross-column rule) but as
pure pandas, so it is importable and unit-tested without a warehouse. It returns
the **clean** rows that pass every rule plus a structured report of how many rows
each rule rejected, so the orchestration layer can fail or warn on the report.

The rules, in the order they are applied (a row is dropped by the first rule it
breaks; the report counts each rule's own rejections without double counting):

1. ``null_key``        — ``station`` or ``date`` is null.
2. ``null_measure``    — any of the four measures is null.
3. ``range``           — a temperature outside ``[-60, 60]`` C or negative precip.
4. ``tmin_gt_tmax``    — the cross-column rule ``tmin_c <= tmax_c`` is violated.
5. ``duplicate``       — a repeated ``(station, date)`` pair (keep the first).
"""

from __future__ import annotations

import pandas as pd

#: Inclusive plausible range for any temperature measure, in degrees Celsius.
TEMP_MIN_C = -60.0
TEMP_MAX_C = 60.0

_MEASURES = ("tmin_c", "tmax_c", "tmean_c", "precip_mm")
_TEMPS = ("tmin_c", "tmax_c", "tmean_c")


def validate_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return ``(clean_df, report)`` after applying the weather DQ rules.

    Parameters
    ----------
    df:
        A tidy weather frame (see :data:`weatherpipe.ingest.WEATHER_COLUMNS`).

    Returns
    -------
    (clean_df, report)
        ``clean_df`` is the subset of rows that pass every rule, with the
        original column order and a reset index. ``report`` is a dict::

            {
              "n_input": int,
              "n_clean": int,
              "n_rejected": int,
              "pct_valid": float,        # n_clean / n_input, 1.0 on empty input
              "rejected": {              # per-rule rejected row counts
                 "null_key": int, "null_measure": int, "range": int,
                 "tmin_gt_tmax": int, "duplicate": int,
              },
            }

        The per-rule counts are disjoint (a row is attributed to the first rule
        it breaks), so they sum to ``n_rejected``.
    """
    work = df.reset_index(drop=True)
    n_input = int(len(work))
    # `alive` marks rows still eligible; each rule rejects only among the alive.
    alive = pd.Series(True, index=work.index)
    rejected: dict[str, int] = {}

    def _reject(mask: pd.Series, name: str) -> None:
        hit = alive & mask
        rejected[name] = int(hit.sum())
        alive.loc[hit] = False

    # 1. null keys
    if n_input:
        null_key = work["station"].isna() | work["date"].isna()
    else:
        null_key = pd.Series(False, index=work.index)
    _reject(null_key, "null_key")

    # 2. null measures
    null_measure = pd.Series(False, index=work.index)
    for col in _MEASURES:
        null_measure |= work[col].isna()
    _reject(null_measure, "null_measure")

    # 3. range: temps outside [-60, 60], or negative precipitation
    out_of_range = pd.Series(False, index=work.index)
    for col in _TEMPS:
        out_of_range |= (work[col] < TEMP_MIN_C) | (work[col] > TEMP_MAX_C)
    out_of_range |= work["precip_mm"] < 0
    _reject(out_of_range, "range")

    # 4. cross-column: tmin must not exceed tmax
    tmin_gt_tmax = work["tmin_c"] > work["tmax_c"]
    _reject(tmin_gt_tmax, "tmin_gt_tmax")

    # 5. duplicate (station, date): keep the first surviving occurrence
    dup = pd.Series(False, index=work.index)
    surviving = work[alive]
    dup_mask = surviving.duplicated(subset=["station", "date"], keep="first")
    dup.loc[surviving.index[dup_mask.to_numpy()]] = True
    _reject(dup, "duplicate")

    clean = work[alive].reset_index(drop=True)
    n_clean = int(len(clean))
    n_rejected = n_input - n_clean
    report = {
        "n_input": n_input,
        "n_clean": n_clean,
        "n_rejected": n_rejected,
        "pct_valid": 1.0 if n_input == 0 else n_clean / n_input,
        "rejected": rejected,
    }
    return clean, report
