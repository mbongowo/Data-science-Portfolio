"""Hand-checked tests for the pandas feature engineering and the time split.

The frames here are small enough that the lag, rolling-mean, and next-day target
values are derived by hand in the comments, so a green test proves the no-look-
ahead construction is correct. numpy / pandas only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlpipe.features import (
    FEATURE_COLUMNS,
    make_features,
    train_test_split_time,
)


def _daily_frame(n: int = 12) -> pd.DataFrame:
    """A tiny deterministic daily frame with known temperature / precip."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    # tmean: 20, 21, 22, ...  precip: 0, 1, 2, 0, 1, 2, ... (period 3)
    tmean = 20.0 + np.arange(n)
    precip = np.array([float(i % 3) for i in range(n)])
    return pd.DataFrame({"date": dates, "tmean_c": tmean, "precip_mm": precip})


def test_make_features_has_expected_columns() -> None:
    """All engineered feature columns plus the target are present."""
    out = make_features(_daily_frame())
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert "rain_tomorrow" in out.columns
    assert "precip_tomorrow" in out.columns


def test_lag_and_rolling_values_hand_checked() -> None:
    """Lag and rolling values match a hand computation on a known row.

    With tmean = [20, 21, 22, 23, 24, 25, ...] and precip = [0,1,2,0,1,2,...],
    the first fully populated row is day index 6 (2022-01-07, tmean 26), the
    earliest day with 3 lags AND a full 7-day rolling window AND a tomorrow.

    For that row (current day = index 6, tmean 26, precip 0):
      tmean_lag1 = tmean[5] = 25, lag2 = tmean[4] = 24, lag3 = tmean[3] = 23.
      precip_lag1 = precip[5] = 2, lag2 = precip[4] = 1, lag3 = precip[3] = 0.
      tmean_roll3 = mean(24, 25, 26) = 25.
      tmean_roll7 = mean(20..26) = 23.
      precip_roll3 = mean(precip[4,5,6]) = mean(1,2,0) = 1.0.
      precip_roll7 = mean(precip[0..6]) = mean(0,1,2,0,1,2,0) = 6/7.
    """
    out = make_features(_daily_frame())
    row = out.loc[out["date"] == pd.Timestamp("2022-01-07")].iloc[0]
    assert row["tmean_lag1"] == pytest.approx(25.0)
    assert row["tmean_lag2"] == pytest.approx(24.0)
    assert row["tmean_lag3"] == pytest.approx(23.0)
    assert row["precip_lag1"] == pytest.approx(2.0)
    assert row["precip_lag2"] == pytest.approx(1.0)
    assert row["precip_lag3"] == pytest.approx(0.0)
    assert row["tmean_roll3"] == pytest.approx(25.0)
    assert row["tmean_roll7"] == pytest.approx(23.0)
    assert row["precip_roll3"] == pytest.approx(1.0)
    assert row["precip_roll7"] == pytest.approx(6.0 / 7.0)


def test_target_is_next_day_rain_no_look_ahead() -> None:
    """rain_tomorrow is precip[t+1] >= 1mm, strictly future relative to row t.

    precip = [0,1,2,0,1,2,...]. For day index 6 (2022-01-07) tomorrow is index 7
    with precip = 1 -> rain_tomorrow = 1. For day index 8 (2022-01-09) tomorrow
    is index 9 with precip = 0 -> rain_tomorrow = 0.
    """
    out = make_features(_daily_frame())
    r7 = out.loc[out["date"] == pd.Timestamp("2022-01-07")].iloc[0]
    r9 = out.loc[out["date"] == pd.Timestamp("2022-01-09")].iloc[0]
    assert r7["precip_tomorrow"] == pytest.approx(1.0)
    assert int(r7["rain_tomorrow"]) == 1
    assert r9["precip_tomorrow"] == pytest.approx(0.0)
    assert int(r9["rain_tomorrow"]) == 0


def test_no_nan_rows_remain() -> None:
    """Rows lacking full lag history or a tomorrow are dropped."""
    out = make_features(_daily_frame())
    assert not out[FEATURE_COLUMNS].isna().any().any()
    # 12 days -> drop first 6 (need 7-day window) and the last (no tomorrow) = 5.
    assert len(out) == 5


def test_make_features_rejects_missing_columns() -> None:
    bad = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=3)})
    with pytest.raises(ValueError):
        make_features(bad)


def test_make_features_rejects_empty() -> None:
    empty = pd.DataFrame({"date": [], "tmean_c": [], "precip_mm": []})
    with pytest.raises(ValueError):
        make_features(empty)


def test_time_split_is_ordered_and_disjoint() -> None:
    """The split is chronological: every train date precedes every test date."""
    out = make_features(_daily_frame(40))
    train_df, test_df = train_test_split_time(out, frac=0.75)
    assert len(train_df) + len(test_df) == len(out)
    assert train_df["date"].max() < test_df["date"].min()
    # Disjoint dates.
    assert set(train_df["date"]).isdisjoint(set(test_df["date"]))


def test_time_split_rejects_bad_frac() -> None:
    out = make_features(_daily_frame(40))
    with pytest.raises(ValueError):
        train_test_split_time(out, frac=0.0)
    with pytest.raises(ValueError):
        train_test_split_time(out, frac=1.0)
