"""Known-answer tests for time-series aggregation.

The toy frame has three documents:

    2019-01-01 (Tue)  score  0.2
    2019-01-01 (Tue)  score  0.4
    2019-01-02 (Wed)  score -0.6

* Daily mean: 2019-01-01 -> (0.2 + 0.4) / 2 = 0.3; 2019-01-02 -> -0.6.
* Weekly mean (pandas W = week ending Sunday): all three fall in the week ending
  2019-01-06, so the single period mean is (0.2 + 0.4 - 0.6) / 3 = 0.0.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sentiment.aggregate import sentiment_timeseries

_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-02"]),
        "score": [0.2, 0.4, -0.6],
    }
)


def test_daily_mean_is_hand_value() -> None:
    out = sentiment_timeseries(_DF, "daily")
    assert list(out["mean_score"].round(10)) == [0.3, -0.6]
    assert list(out["n"]) == [2, 1]
    assert list(out["period"].dt.strftime("%Y-%m-%d")) == [
        "2019-01-01",
        "2019-01-02",
    ]


def test_weekly_mean_is_hand_value() -> None:
    out = sentiment_timeseries(_DF, "weekly")
    # One week, ending Sunday 2019-01-06.
    assert len(out) == 1
    assert out.loc[0, "mean_score"] == pytest.approx(0.0, abs=1e-12)
    assert out.loc[0, "n"] == 3
    assert out.loc[0, "period"].strftime("%Y-%m-%d") == "2019-01-06"


def test_unknown_freq_raises() -> None:
    with pytest.raises(ValueError):
        sentiment_timeseries(_DF, "monthly")


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        sentiment_timeseries(pd.DataFrame({"date": [], "value": []}), "daily")
