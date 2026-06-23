"""Feature engineering for the rain-day task, in pure pandas.

This is the bridge from the weather data-engineering project's daily frame to a
supervised learning matrix. :func:`make_features` turns one row per day into lag
and rolling-window features plus a seasonal (day-of-year) encoding, and attaches
the binary target ``rain_tomorrow`` (next-day precipitation reaches the rain-day
threshold).

**No look-ahead.** Every feature for day *t* is built only from information known
*by the end of day t*:

* lag features (``tmean_lag1..3``, ``precip_lag1..3``) shift past values forward;
* rolling means (``tmean_roll3``, ``precip_roll7`` ...) are taken over windows
  ending at *t* (no centering);
* the target is ``precip`` of day *t+1*, obtained by a backward shift, so it is
  strictly future relative to the features.

Rows that cannot be fully populated (the first few days, which lack enough lag
history, and the final day, which has no tomorrow) are dropped. The companion
:func:`train_test_split_time` splits in time order so the test set is always the
*later* data — a model is never evaluated on days that precede its training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A day is a "rain day" when precipitation reaches this many millimetres.
RAIN_DAY_MM = 1.0

#: Lag depths and rolling windows used by :func:`make_features`.
LAGS = (1, 2, 3)
ROLL_WINDOWS = (3, 7)

#: The engineered feature columns, in a fixed order (matches serving payloads).
FEATURE_COLUMNS = [
    "tmean_lag1",
    "tmean_lag2",
    "tmean_lag3",
    "precip_lag1",
    "precip_lag2",
    "precip_lag3",
    "tmean_roll3",
    "tmean_roll7",
    "precip_roll3",
    "precip_roll7",
    "doy_sin",
    "doy_cos",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag / rolling / seasonal features and the ``rain_tomorrow`` target.

    Parameters
    ----------
    df:
        Daily weather frame with at least ``date``, ``tmean_c`` and ``precip_mm``
        columns (a ``station`` column is preserved if present). Rows are sorted by
        date inside this function.

    Returns
    -------
    pandas.DataFrame
        One row per usable day with the columns in :data:`FEATURE_COLUMNS`, the
        original ``date`` (and ``station`` if given), the next-day
        ``precip_tomorrow``, and the binary ``rain_tomorrow`` target. Rows
        without full lag history or without a following day are dropped.

    Raises
    ------
    ValueError
        If required columns are missing or the frame is empty.
    """
    required = {"date", "tmean_c", "precip_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"make_features needs columns {sorted(missing)} (missing).")
    if df.empty:
        raise ValueError("make_features got an empty frame.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)

    # Lag features: value from N days ago, known by the end of day t.
    for lag in LAGS:
        work[f"tmean_lag{lag}"] = work["tmean_c"].shift(lag)
        work[f"precip_lag{lag}"] = work["precip_mm"].shift(lag)

    # Rolling means over windows ending at t (shift(1) so day t itself is excluded
    # only where that would peek; here windows include t, which is allowed since
    # t is observed by end of day). Use min_periods == window to avoid partials.
    for window in ROLL_WINDOWS:
        work[f"tmean_roll{window}"] = (
            work["tmean_c"].rolling(window, min_periods=window).mean()
        )
        work[f"precip_roll{window}"] = (
            work["precip_mm"].rolling(window, min_periods=window).mean()
        )

    # Day-of-year seasonal encoding (continuous, wraps at year end).
    doy = work["date"].dt.dayofyear.to_numpy(dtype=float)
    work["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    work["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Target: tomorrow's precipitation (strictly future) and the rain-day flag.
    work["precip_tomorrow"] = work["precip_mm"].shift(-1)
    work["rain_tomorrow"] = (work["precip_tomorrow"] >= RAIN_DAY_MM).astype(int)

    keep = ["date"]
    if "station" in work.columns:
        keep.append("station")
    keep += FEATURE_COLUMNS + ["precip_tomorrow", "rain_tomorrow"]

    out = work[keep].dropna().reset_index(drop=True)
    out["rain_tomorrow"] = out["rain_tomorrow"].astype(int)
    return out


def train_test_split_time(
    df: pd.DataFrame, frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature frame in **time order** into train / test.

    The frame is sorted by ``date`` and the first ``frac`` of the rows become the
    training set, the remainder the test set. Because the cut is chronological,
    the test set is always strictly later than the train set — the only honest
    way to estimate how the model will do on future days.

    Parameters
    ----------
    df:
        Output of :func:`make_features` (must contain a ``date`` column).
    frac:
        Fraction of rows assigned to training, in ``(0, 1)``.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train_df, test_df)``, each with a reset index. They are disjoint and,
        concatenated in order, reconstruct the time-sorted frame.

    Raises
    ------
    ValueError
        If ``frac`` is not in ``(0, 1)`` or ``date`` is missing.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in the open interval (0, 1).")
    if "date" not in df.columns:
        raise ValueError("train_test_split_time needs a 'date' column.")

    ordered = df.sort_values("date").reset_index(drop=True)
    cut = int(len(ordered) * frac)
    train_df = ordered.iloc[:cut].reset_index(drop=True)
    test_df = ordered.iloc[cut:].reset_index(drop=True)
    return train_df, test_df
