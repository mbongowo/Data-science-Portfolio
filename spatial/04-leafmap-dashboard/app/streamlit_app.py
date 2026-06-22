"""Streamlit + leafmap dashboard: which populated places are farthest from a clinic?

Run locally with::

    streamlit run app/streamlit_app.py

The app loads bundled synthetic Cameroon sample data by default (or your own
uploaded CSVs), computes each place's straight-line (haversine) distance to the
nearest health facility with the reused ``clinicaccess`` core, and draws an
interactive leafmap map: facilities as markers, places coloured by their
distance band, and the places beyond a chosen "underserved" threshold
highlighted. A metric row and a table of the farthest-N underserved places sit
alongside.

Distances are straight-line, not road travel time -- this is a fast screening
tool. The sibling ``access-to-care`` project computes rigorous road-network
travel time; the two complement each other.
"""

from __future__ import annotations

# Streamlit Community Cloud runs this file directly, so only this file's
# directory is on sys.path and the `clinicaccess` package (under ../src) is not
# importable. Put the project's src/ on the path first.
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

import pandas as pd
import streamlit as st

from clinicaccess.access import coverage_stats, distance_bins, farthest_places, nearest_facility

SAMPLE_DIR = _Path(__file__).resolve().parent / "sample_data"
DEFAULT_PLACES = SAMPLE_DIR / "cameroon_places.csv"
DEFAULT_FACILITIES = SAMPLE_DIR / "cameroon_facilities.csv"

THRESHOLDS_KM = [5, 10, 25]
BIN_EDGES = [5, 10, 25]
# Colour-blind-friendly graduated ramp, near -> far.
BIN_COLOURS = {
    "0-5 km": "#1a9850",
    "5-10 km": "#91cf60",
    "10-25 km": "#fee08b",
    "25+ km": "#d73027",
}

st.set_page_config(
    page_title="Clinic-access dashboard (Cameroon)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _read_csv(data: bytes | str) -> pd.DataFrame:
    return pd.read_csv(data)


def _load_table(uploaded, default_path: _Path, kind: str) -> pd.DataFrame | None:
    """Read an uploaded CSV if present, else the bundled sample; validate columns."""
    try:
        df = _read_csv(uploaded) if uploaded is not None else _read_csv(str(default_path))
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        st.warning(f"Could not read the {kind} CSV: {exc}")
        return None
    missing = [c for c in ("lat", "lon") if c not in df.columns]
    if missing:
        st.warning(f"The {kind} CSV is missing required columns: {missing}")
        return None
    return df


def main() -> None:
    st.title("🏥 Clinic-access dashboard — Cameroon")
    st.caption(
        "Which populated places are farthest from a health facility? Distances here "
        "are **straight-line (haversine)**, not road travel time — a fast screening "
        "tool, not a routing study. For rigorous road-network travel time see the "
        "sibling `access-to-care` project."
    )

    with st.sidebar:
        st.header("Data")
        st.markdown(
            "Using the **bundled synthetic Cameroon sample** by default. "
            "Upload your own CSVs to screen another area."
        )
        places_file = st.file_uploader(
            "Places CSV (lat, lon, population, name)", type="csv", key="places"
        )
        facilities_file = st.file_uploader(
            "Facilities CSV (lat, lon, name)", type="csv", key="facilities"
        )
        st.header("Threshold")
        threshold_km = st.slider(
            "Underserved if nearest facility is beyond (km)",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
        )
        farthest_n = st.slider("How many farthest places to list", 5, 50, 10, step=5)

    places = _load_table(places_file, DEFAULT_PLACES, "places")
    facilities = _load_table(facilities_file, DEFAULT_FACILITIES, "facilities")
    if places is None or facilities is None:
        st.stop()
    if "population" not in places.columns:
        places = places.assign(population=1.0)

    try:
        access = nearest_facility(places, facilities)
    except (KeyError, ValueError) as exc:
        st.warning(f"Could not compute nearest-facility distances: {exc}")
        st.stop()

    access["band"] = distance_bins(access["nearest_km"], BIN_EDGES).astype(str)
    access["underserved"] = access["nearest_km"] > threshold_km
    stats = coverage_stats(access["nearest_km"], access["population"], THRESHOLDS_KM)
    share_within = (
        float(
            (access.loc[access["nearest_km"] <= threshold_km, "population"].sum())
            / access["population"].sum()
        )
        if access["population"].sum() > 0
        else 0.0
    )

    # --- metric row ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Places", f"{len(access):,}")
    c2.metric("Mean nearest", f"{access['nearest_km'].mean():.1f} km")
    c3.metric("Median nearest", f"{access['nearest_km'].median():.1f} km")
    c4.metric(f"Pop. within {threshold_km} km", f"{share_within:.0%}")

    # --- map ----------------------------------------------------------------
    import leafmap.foliumap as leafmap

    center_lat = float(places["lat"].mean())
    center_lon = float(places["lon"].mean())
    m = leafmap.Map(center=(center_lat, center_lon), zoom=6, draw_control=False)
    m.add_basemap("CartoDB.Positron")

    for _, row in access.iterrows():
        colour = BIN_COLOURS.get(row["band"], "#777777")
        underserved = bool(row["underserved"])
        name = row.get("name", "place")
        m.add_marker(
            location=(float(row["lat"]), float(row["lon"])),
            popup=(
                f"<b>{name}</b><br>nearest facility: {row['nearest_km']:.1f} km"
                f"<br>population: {int(row['population']):,}"
                f"<br>band: {row['band']}"
            ),
            icon=None,
            radius=8 if underserved else 5,
            color="#000000" if underserved else colour,
            weight=2 if underserved else 1,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
        )

    for _, row in facilities.iterrows():
        fname = row.get("name", "facility")
        m.add_marker(
            location=(float(row["lat"]), float(row["lon"])),
            popup=f"<b>{fname}</b><br>health facility",
            icon={"color": "blue", "icon": "plus", "prefix": "fa"},
        )

    legend = {**BIN_COLOURS, "facility (marker)": "#2c7fb8"}
    m.add_legend(title="Distance to nearest facility", legend_dict=legend)
    m.to_streamlit(height=560)

    st.caption(
        "Circles are populated places coloured by straight-line distance to the "
        "nearest facility; black-ringed larger circles are beyond the chosen "
        f"{threshold_km} km threshold. Blue plus markers are health facilities."
    )

    # --- farthest table -----------------------------------------------------
    st.subheader(f"Farthest {farthest_n} underserved places")
    underserved = access[access["underserved"]]
    table = farthest_places(underserved if len(underserved) else access, n=farthest_n)
    show_cols = [
        c for c in ("name", "lat", "lon", "population", "nearest_km", "band") if c in table.columns
    ]
    st.dataframe(
        table[show_cols]
        .reset_index(drop=True)
        .style.format({"nearest_km": "{:.1f}", "lat": "{:.3f}", "lon": "{:.3f}"}),
        use_container_width=True,
    )

    with st.expander("Coverage detail"):
        st.json(stats)


if __name__ == "__main__":
    main()
