"""Streamlit dashboard for the air-quality alerts and hourly AQI (heavy path).

Runs in the docker-compose ``dashboard`` service. It reads the DuckDB sink the
processor writes (falling back to the demo's ``outputs/`` CSVs when no live
database is present) and shows the recent alerts and per-station AQI. This module
imports ``streamlit`` / ``pandas`` at the top because it is only ever launched by
``streamlit run``; it is never imported by the test suite or the package
``__init__``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_SINK_DB = Path("/data/aqstream.duckdb")
_OUTPUTS = Path("outputs")


def _load_alerts() -> pd.DataFrame:
    if _SINK_DB.exists():
        import duckdb

        con = duckdb.connect(str(_SINK_DB))
        df = con.execute("SELECT * FROM alerts ORDER BY ts DESC").fetchdf()
        con.close()
        return df
    csv = _OUTPUTS / "alerts.csv"
    return pd.read_csv(csv) if csv.exists() else pd.DataFrame()


def _load_hourly() -> pd.DataFrame:
    csv = _OUTPUTS / "hourly_aqi.csv"
    return pd.read_csv(csv) if csv.exists() else pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Air-quality alerts — Cameroon", layout="wide")
    st.title("Air-quality streaming & alerting — Cameroon cities")

    alerts = _load_alerts()
    st.subheader(f"Fired alerts ({len(alerts)})")
    if not alerts.empty:
        st.dataframe(alerts, use_container_width=True)
    else:
        st.info("No alerts yet — run the demo or start the live producer.")

    hourly = _load_hourly()
    if not hourly.empty:
        st.subheader("Hourly AQI by station")
        pivot = hourly.pivot_table(
            index="window_start", columns="station", values="aqi"
        )
        st.line_chart(pivot)


if __name__ == "__main__":
    main()
