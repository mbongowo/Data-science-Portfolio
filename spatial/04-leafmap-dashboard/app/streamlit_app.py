"""Streamlit + leafmap dashboard: which populated places are farthest from a clinic?

Run locally with::

    streamlit run app/streamlit_app.py

The app loads bundled real Cameroon data from OpenStreetMap by default (or your own
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
            "Using **real Cameroon health facilities and populated places from "
            "OpenStreetMap** by default (© OpenStreetMap contributors, ODbL). "
            "Upload your own CSVs to screen another area."
        )
        places_file = st.file_uploader(
            "Places CSV (lat, lon, population, name)", type="csv", key="places"
        )
        facilities_file = st.file_uploader(
            "Facilities CSV (lat, lon, name)", type="csv", key="facilities"
        )

    places = _load_table(places_file, DEFAULT_PLACES, "places")
    facilities = _load_table(facilities_file, DEFAULT_FACILITIES, "facilities")
    if places is None or facilities is None:
        st.stop()
    if "population" not in places.columns:
        places = places.assign(population=1.0)

    # --- area drill-down: Region -> Division -> Sub-division ---------------- #
    area_label = "All Cameroon"
    with st.sidebar:
        if "region" in places.columns:
            st.header("Area")
            regions = ["All Cameroon"] + sorted({r for r in places["region"].dropna() if r})
            region_sel = st.selectbox("Region", regions, index=0)
            if region_sel != "All Cameroon":
                places = places[places["region"] == region_sel]
                area_label = region_sel
                if "division" in places.columns:
                    divs = ["All divisions"] + sorted({d for d in places["division"].dropna() if d})
                    div_sel = st.selectbox("Division", divs, index=0)
                    if div_sel != "All divisions":
                        places = places[places["division"] == div_sel]
                        area_label = f"{div_sel} division, {region_sel}"
                        if "subdivision" in places.columns:
                            subs = ["All sub-divisions"] + sorted(
                                {s for s in places["subdivision"].dropna() if s}
                            )
                            sub_sel = st.selectbox("Sub-division", subs, index=0)
                            if sub_sel != "All sub-divisions":
                                places = places[places["subdivision"] == sub_sel]
                                area_label = f"{sub_sel}, {div_sel}"
            st.caption(
                "Sub-division (arrondissement) is the lowest official boundary "
                "level in Cameroon — there is no separate township layer."
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

    if places.empty:
        st.warning("No populated places in the selected area.")
        st.stop()

    # Distances are computed against ALL facilities, so a place near an admin
    # boundary still finds the nearest facility just across it.
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
    if area_label != "All Cameroon":
        st.caption(f"📍 Showing **{area_label}** — {len(access):,} populated places.")

    # --- map ----------------------------------------------------------------
    import json as _json
    from pathlib import Path as _MapPath

    import folium
    import leafmap.foliumap as leafmap
    from folium.plugins import FastMarkerCluster

    center_lat = float(places["lat"].mean())
    center_lon = float(places["lon"].mean())
    m = leafmap.Map(center=(center_lat, center_lon), zoom=6, draw_control=False)
    m.add_basemap("CartoDB.Positron")

    # Administrative boundaries as toggleable outline layers (regions + divisions).
    # Sub-divisions drive the drill-down filter but are not drawn — all 360
    # polygons would bloat the map; use the sidebar selector to focus on one.
    _bdir = _MapPath(__file__).resolve().parent / "boundaries"
    for fn, label, weight, line_colour, show in (
        ("cmr_adm2.geojson", "Divisions", 1.0, "#999999", True),
        ("cmr_adm1.geojson", "Regions", 2.5, "#333333", True),
    ):
        fpath = _bdir / fn
        if fpath.exists():
            _fg = folium.FeatureGroup(name=label, show=show).add_to(m)
            folium.GeoJson(
                _json.loads(fpath.read_text(encoding="utf-8")),
                style_function=lambda _f, c=line_colour, w=weight: {
                    "color": c, "weight": w, "fill": False, "opacity": 0.7,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["name"], aliases=[label.rstrip("s") + ":"]
                ),
            ).add_to(_fg)

    # Populated places as graduated circle markers (folium.CircleMarker, not the
    # default pin) coloured by distance band; underserved places get a black ring.
    places_fg = folium.FeatureGroup(name="Populated places", show=True).add_to(m)
    for _, row in access.iterrows():
        colour = BIN_COLOURS.get(row["band"], "#777777")
        underserved = bool(row["underserved"])
        name = row.get("name", "place")
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=7 if underserved else 4,
            color="#000000" if underserved else colour,
            weight=2 if underserved else 1,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            popup=(
                f"<b>{name}</b><br>nearest facility: {row['nearest_km']:.1f} km"
                f"<br>population: {int(row['population']):,}"
                f"<br>band: {row['band']}"
            ),
        ).add_to(places_fg)

    # Facilities to DISPLAY: all for the country view, else those near the
    # selected area (the distance metric above already used every facility).
    if area_label != "All Cameroon" and len(access):
        pad = 0.4
        fac_display = facilities[
            facilities["lat"].between(access["lat"].min() - pad, access["lat"].max() + pad)
            & facilities["lon"].between(access["lon"].min() - pad, access["lon"].max() + pad)
        ]
    else:
        fac_display = facilities

    # Health facilities via FastMarkerCluster — compact rendering for thousands
    # of points (per-facility names are omitted to keep the page light).
    FastMarkerCluster(
        data=fac_display[["lat", "lon"]].to_numpy().tolist(),
        name="Health facilities",
    ).add_to(m)

    if area_label != "All Cameroon" and len(access):
        m.fit_bounds(
            [
                [float(access["lat"].min()), float(access["lon"].min())],
                [float(access["lat"].max()), float(access["lon"].max())],
            ]
        )

    folium.LayerControl(collapsed=False).add_to(m)
    legend = {**BIN_COLOURS, "facility (clustered)": "#2c7fb8"}
    m.add_legend(title="Distance to nearest facility", legend_dict=legend)
    m.to_streamlit(height=600)

    st.caption(
        "Circles are populated places coloured by straight-line distance to the "
        "nearest facility; black-ringed circles are beyond the chosen "
        f"{threshold_km} km threshold. Clustered blue markers are health "
        "facilities. Use the layer control (top-right of the map) to toggle "
        "regions, divisions, places, and facilities."
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
