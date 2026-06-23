"""Drift monitoring behind a lazy Evidently import.

This is the dashboard face of the drift core. :func:`drift_dashboard` builds an
Evidently data-drift report (an interactive HTML page with per-feature
distributions and drift verdicts) so a reviewer can *see* the shift, not just
read a number.

Evidently is optional. The runnable, dependency-free fallback is
:func:`mlpipe.drift.feature_drift_report`, which computes the same PSI / KS
signals in pure numpy and is what the demo and the test suite use. Use Evidently
when you want the rich report; use the numpy core when you want something that
runs anywhere (including CI) with no extra install.

``evidently`` is imported lazily inside the function, so importing this module is
cheap and the test suite never pulls it in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def drift_dashboard(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    out_html: str | Path = "outputs/drift_report.html",
) -> str:
    """Build an Evidently data-drift report and save it as HTML.

    Parameters
    ----------
    reference_df, current_df:
        The baseline and the recent data, same columns.
    out_html:
        Where to write the HTML report.

    Returns
    -------
    str
        The path to the written HTML file.

    Notes
    -----
    For a dependency-free equivalent that returns a tidy table (and powers the
    demo and tests), use :func:`mlpipe.drift.feature_drift_report`.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    return str(out)


def drift_summary(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> dict[str, Any]:
    """Dependency-free drift summary built on :mod:`mlpipe.drift`.

    A convenience that returns the numpy-core report's ``summary`` dict, so the
    monitoring step has a no-install path that still answers "did anything
    drift?".
    """
    from mlpipe.drift import feature_drift_report

    report = feature_drift_report(reference_df, current_df)
    return dict(report.attrs["summary"])
