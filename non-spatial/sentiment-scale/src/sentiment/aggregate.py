"""Aggregate per-document sentiment scores into a time series.

Given a frame with a ``date`` column and a ``score`` column, collapse the scores
into a mean per calendar period (daily or weekly). This is the step that turns a
pile of scored documents into the "sentiment over time" series the README talks
about.

The implementation is plain pandas so it has no dependency beyond numpy/pandas
and is covered by a known-answer test on a tiny frame.
"""

from __future__ import annotations

import pandas as pd

#: Map the config vocabulary to pandas offset aliases.
_FREQ_ALIASES: dict[str, str] = {
    "daily": "D",
    "weekly": "W",
}


def sentiment_timeseries(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """Return the mean sentiment score per period.

    Parameters
    ----------
    df:
        A frame with a ``date`` column (datetime-like) and a numeric ``score``
        column. One row per scored document.
    freq:
        ``"daily"`` or ``"weekly"`` (pandas ``D`` / ``W`` resampling). Anything
        else raises.

    Returns
    -------
    pandas.DataFrame
        A frame with columns ``period``, ``mean_score`` and ``n`` (the document
        count in that period), sorted by ``period``. ``W`` periods are labelled
        by the week-ending date, following pandas' convention.

    Raises
    ------
    ValueError
        If ``freq`` is unknown or a required column is missing.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-02"]),
    ...         "score": [0.2, 0.4, -0.6],
    ...     }
    ... )
    >>> out = sentiment_timeseries(df, "daily")
    >>> list(out["mean_score"].round(6))
    [0.3, -0.6]
    """
    if freq not in _FREQ_ALIASES:
        raise ValueError(
            f"Unknown freq {freq!r}; expected one of {sorted(_FREQ_ALIASES)}."
        )
    for col in ("date", "score"):
        if col not in df.columns:
            raise ValueError(f"Input frame is missing required column {col!r}.")

    alias = _FREQ_ALIASES[freq]
    dates = pd.to_datetime(df["date"])
    grouped = (
        df.assign(_period=dates.dt.to_period(alias).dt.to_timestamp(how="end"))
        .groupby("_period")["score"]
        .agg(mean_score="mean", n="size")
        .reset_index()
        .rename(columns={"_period": "period"})
        .sort_values("period")
        .reset_index(drop=True)
    )
    # Normalise the period column to a plain date (drop the end-of-day time that
    # to_timestamp(how="end") introduces) so daily/weekly labels read cleanly.
    grouped["period"] = grouped["period"].dt.normalize()
    return grouped
