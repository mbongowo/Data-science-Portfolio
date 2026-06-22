"""Technical indicators on a price (or return) series.

Pure numpy/pandas, no third-party dependency, so these are covered by
known-answer tests. Each function takes and returns a pandas ``Series`` (or
accepts an array-like that is coerced to one) and preserves the input index, so
indicator outputs line up bar-for-bar with the prices they came from.

All windows are *trailing*: the value at bar ``t`` uses only bars up to and
including ``t``. That matters for the backtest — an indicator must never see
the future — and it is why the leading bars (before a full window exists) are
``NaN`` rather than back-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x: pd.Series | np.ndarray | list[float]) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(np.asarray(x, dtype=float))


def sma(x: pd.Series | np.ndarray | list[float], n: int) -> pd.Series:
    """Simple moving average over a trailing window of ``n`` bars.

    The value at bar ``t`` is the unweighted mean of ``x[t-n+1 .. t]``. The
    first ``n - 1`` positions are ``NaN`` because no full window exists yet.
    """
    if n < 1:
        raise ValueError("Window n must be >= 1.")
    return _as_series(x).rolling(window=n, min_periods=n).mean()


def ema(
    x: pd.Series | np.ndarray | list[float],
    n: int,
    *,
    adjust: bool = False,
) -> pd.Series:
    """Exponential moving average with span ``n`` (smoothing ``2/(n+1)``).

    With ``adjust=False`` (default) this is the recursive form
    ``e_t = alpha * x_t + (1 - alpha) * e_{t-1}`` seeded from the first value,
    which is the convention trading systems use. Pass ``adjust=True`` for
    pandas' bias-corrected weighting.
    """
    if n < 1:
        raise ValueError("Span n must be >= 1.")
    return _as_series(x).ewm(span=n, adjust=adjust).mean()


def rsi(x: pd.Series | np.ndarray | list[float], n: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index over ``n`` bars.

    Gains and losses of the bar-to-bar price change are smoothed with Wilder's
    moving average (an EMA with ``alpha = 1/n``). ``RSI = 100 - 100/(1 + RS)``
    where ``RS`` is the average gain over the average loss.

    Edge behaviour: a strictly increasing series has no losses, so the average
    loss is zero, ``RS`` is infinite, and ``RSI = 100`` exactly — the
    known-answer the test pins. The symmetric all-down case gives ``RSI = 0``.
    """
    if n < 1:
        raise ValueError("Window n must be >= 1.")
    s = _as_series(x)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing == EMA with alpha = 1/n.
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> rs == inf -> out == 100 (handled); but 0/0 -> NaN -> 100.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    # The very first delta is NaN, so the first RSI value is undefined.
    out.iloc[0] = np.nan
    return out


def rolling_vol(
    returns: pd.Series | np.ndarray | list[float],
    n: int,
    *,
    ddof: int = 0,
) -> pd.Series:
    """Rolling standard deviation of returns over a trailing ``n``-bar window.

    Uses the population standard deviation (``ddof=0``) by default so the
    known-answer test can hand-derive it. Pass ``ddof=1`` for the sample
    standard deviation. The first ``n - 1`` positions are ``NaN``.
    """
    if n < 1:
        raise ValueError("Window n must be >= 1.")
    return _as_series(returns).rolling(window=n, min_periods=n).std(ddof=ddof)


def macd(
    x: pd.Series | np.ndarray | list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    Returns a :class:`pandas.DataFrame` with three trailing columns:

    * ``macd``     — ``ema(x, fast) - ema(x, slow)`` (recursive EMA, ``adjust=False``).
    * ``signal``   — ``ema(macd, signal)`` (the signal line).
    * ``histogram``— ``macd - signal``.

    All three are strictly trailing: every value at bar ``t`` uses only bars up
    to ``t``, because each leg is the recursive EMA seeded from the first value.

    A constant series has zero EMA dispersion, so the MACD line, signal line and
    histogram are all ``0.0`` everywhere — the known-answer the test pins.
    """
    if fast < 1 or slow < 1 or signal < 1:
        raise ValueError("fast, slow and signal must all be >= 1.")
    if fast >= slow:
        raise ValueError("fast span must be strictly less than slow span.")
    s = _as_series(x)
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=s.index,
    )


def bollinger_bands(
    x: pd.Series | np.ndarray | list[float],
    n: int = 20,
    k: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands over a trailing ``n``-bar window.

    Returns a :class:`pandas.DataFrame` with three columns:

    * ``middle`` — the simple moving average, ``sma(x, n)``.
    * ``upper``  — ``middle + k * sigma``.
    * ``lower``  — ``middle - k * sigma``.

    ``sigma`` is the *population* rolling standard deviation (``ddof=0``) so the
    band width is hand-derivable. The first ``n - 1`` positions are ``NaN``.

    A constant series has ``sigma = 0``, so all three bands collapse onto the
    price — the known-answer the test pins.
    """
    if n < 1:
        raise ValueError("Window n must be >= 1.")
    s = _as_series(x)
    middle = s.rolling(window=n, min_periods=n).mean()
    sigma = s.rolling(window=n, min_periods=n).std(ddof=0)
    upper = middle + k * sigma
    lower = middle - k * sigma
    return pd.DataFrame(
        {"middle": middle, "upper": upper, "lower": lower}, index=s.index
    )


def atr(
    high: pd.Series | np.ndarray | list[float],
    low: pd.Series | np.ndarray | list[float],
    close: pd.Series | np.ndarray | list[float],
    n: int = 14,
) -> pd.Series:
    """Average True Range (Wilder) over ``n`` bars.

    The *true range* of bar ``t`` is

        ``max(high - low, |high - prev_close|, |low - prev_close|)``

    using the previous bar's close. On the first bar there is no prior close, so
    the true range falls back to ``high - low``. The ATR is Wilder's moving
    average of the true range (an EMA with ``alpha = 1/n``), exactly as for RSI.

    Strictly trailing: bar ``t``'s value uses only bars up to ``t``.
    """
    if n < 1:
        raise ValueError("Window n must be >= 1.")
    h = _as_series(high)
    low_s = _as_series(low)
    c = _as_series(close)
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            h - low_s,
            (h - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # First bar: no prior close -> true range is just the bar's high-low.
    tr.iloc[0] = float(h.iloc[0] - low_s.iloc[0])
    # Wilder smoothing == EMA with alpha = 1/n.
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()
