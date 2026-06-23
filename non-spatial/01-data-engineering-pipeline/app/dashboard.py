"""A small Streamlit dashboard over the weather marts.

Reads the ``monthly_weather`` mart (from DuckDB if a warehouse exists, otherwise
from the demo's ``outputs/monthly_summary.csv``) and charts monthly temperature
and precipitation per station. Every heavy import (streamlit, duckdb, pandas) is
lazy / local to ``main`` so importing this module costs nothing and the test
suite never pulls in Streamlit.

Run it:

    streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path


def load_monthly(
    db_path: str = "data/warehouse.duckdb",
    csv_fallback: str = "outputs/monthly_summary.csv",
):
    """Load the monthly mart from DuckDB, or fall back to the demo CSV."""
    import pandas as pd

    if Path(db_path).exists():
        try:
            from weatherpipe.warehouse import query

            return query(db_path, "select * from marts.monthly_weather")
        except Exception:  # pragma: no cover - dashboard convenience
            pass
    return pd.read_csv(csv_fallback)


def main() -> None:  # pragma: no cover - exercised by Streamlit, not pytest
    import streamlit as st

    st.set_page_config(page_title="Cameroon weather", layout="wide")
    st.title("Cameroon weather — monthly marts")
    st.caption(
        "Open-Meteo history -> partitioned lake -> warehouse -> dbt marts. "
        "Reading the monthly_weather mart (DuckDB if present, else the demo CSV)."
    )

    monthly = load_monthly()
    monthly["period"] = (
        monthly["year"].astype(int).astype(str)
        + "-"
        + monthly["month"].astype(int).astype(str).str.zfill(2)
    )

    stations = sorted(monthly["station"].unique())
    chosen = st.multiselect("Stations", stations, default=stations)
    view = monthly[monthly["station"].isin(chosen)]

    st.subheader("Mean monthly temperature (C)")
    temp = view.pivot_table(index="period", columns="station", values="tmean_mean")
    st.line_chart(temp)

    st.subheader("Total monthly precipitation (mm)")
    precip = view.pivot_table(
        index="period", columns="station", values="precip_total_mm"
    )
    st.bar_chart(precip)

    st.subheader("Monthly mart")
    st.dataframe(view.sort_values(["station", "year", "month"]))


if __name__ == "__main__":  # pragma: no cover
    main()
