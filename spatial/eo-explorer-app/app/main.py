"""Streamlit entry point for the EO Explorer app.

Run locally with::

    streamlit run app/main.py

Workflow
--------
1. The user draws an area of interest on the map (rectangle or polygon tool),
   optionally jumping there first with the on-map place-name search.
2. The sidebar offers a date picker and a category-grouped index selector (34
   indices across Vegetation / Water / Soil / Built-up / Snow / Fire).
3. On "Load", the app finds the least-cloudy Sentinel-2 L2A scene near the date,
   loads the needed bands, computes the index with the reused ``eo-monitor``
   functions, and draws it on the map as a coloured overlay with a legend.

The drawn geometry is read back from the map with ``st_folium``. STAC queries and
loads are cached by AOI + date + index, and the computed overlay is kept in
session state so it stays on the map across reruns.
"""

from __future__ import annotations

# When a host runs this file directly (Streamlit Community Cloud runs
# app/main.py), only this file's directory is on sys.path, so the `app` package
# is not importable. Put the project root (the parent of app/) on the path.
import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parent.parent)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import datetime as _dt

import streamlit as st

from app import render, stac

st.set_page_config(
    page_title="EO Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _cached_find_and_load(bbox: tuple[float, ...], date: str, index: str):
    """Find the best scene and load its bands, cached by (bbox, date, index)."""
    stac.cache_key(bbox, date, index)  # documents the cache identity
    item = stac.find_scene(bbox, date)
    if item is None:
        return None, None
    dataset = stac.load_scene(item, bbox, index)
    return stac.scene_metadata(item), dataset


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #


def _sidebar() -> tuple[str, _dt.date, float]:
    st.sidebar.title("EO Explorer")
    st.sidebar.markdown(
        "Draw an area on the map, pick a **date** and an **index**, then load "
        "live Sentinel-2 imagery from the Earth Search STAC catalogue."
    )

    by_cat = render.list_indices_by_category()
    category = st.sidebar.selectbox(
        "Index category",
        options=list(by_cat),
        help="Indices are grouped by what they measure.",
    )
    index = st.sidebar.selectbox(
        "Spectral index",
        options=by_cat[category],
        format_func=lambda key: render.INDEX_REGISTRY[key].name,
        help="Index functions are reused from the eo-monitor package.",
    )
    st.sidebar.caption(render.INDEX_REGISTRY[index].description)

    today = _dt.date.today()
    date = st.sidebar.date_input(
        "Target date",
        value=today - _dt.timedelta(days=14),
        min_value=_dt.date(2015, 6, 23),  # Sentinel-2 archive start
        max_value=today,
        help="The app searches +/- 10 days around this date for the least-cloudy scene.",
    )

    max_area = st.sidebar.slider(
        "Max AOI area (km^2)",
        min_value=100,
        max_value=5000,
        value=int(stac.DEFAULT_MAX_AREA_KM2),
        step=100,
        help="Larger areas take longer to load and are rejected above this limit.",
    )

    if not render.EO_MONITOR_AVAILABLE:
        st.sidebar.error(
            "eo-monitor is not installed, so index computation is disabled. "
            "Install it with `pip install -e ../eo-monitor`."
        )

    return index, date, float(max_area)


# --------------------------------------------------------------------------- #
# Map + draw capture
# --------------------------------------------------------------------------- #


def _build_map(overlay: dict | None):
    """Build the folium map.

    Layers: an Esri "World Imagery" satellite basemap (default) and a labelled
    OpenStreetMap basemap for orientation, a place-name search box (Geocoder, via
    OSM Nominatim, no API key), draw tools, a layer switcher, and any saved
    overlay.
    """
    import folium
    from folium.plugins import Draw, Geocoder

    fmap = folium.Map(location=[10.0, 15.0], zoom_start=3, control_scale=True, tiles=None)

    # Imagery first so it is the default base layer; OSM gives place/road labels
    # so the user can tell where they are. Both are switchable via LayerControl.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles (c) Esri",
        name="Esri World Imagery",
        control=True,
        show=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (labels)",
        control=True,
        show=False,
    ).add_to(fmap)

    # Place-name search: type e.g. "Nairobi" and the map flies there. Uses
    # Leaflet-Control-Geocoder against OSM Nominatim, so no API key is needed.
    Geocoder(collapsed=False, add_marker=True, placeholder="Search for a place...").add_to(fmap)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
    ).add_to(fmap)

    if overlay is not None:
        folium.raster_layers.ImageOverlay(
            image=overlay["image_uri"],
            bounds=overlay["bounds"],
            opacity=overlay.get("opacity", 0.8),
            name=overlay["name"],
            interactive=False,
        ).add_to(fmap)
        render.add_overlay_legend(fmap, overlay)
        fmap.fit_bounds(overlay["bounds"])

    # Added last so it sees every layer (both basemaps + any overlay).
    folium.LayerControl(collapsed=True).add_to(fmap)

    return fmap


def _latest_drawing(map_state: dict | None):
    """Return the most recent drawn GeoJSON feature from st_folium output."""
    if not map_state:
        return None
    drawing = map_state.get("last_active_drawing")
    if drawing:
        return drawing
    all_drawings = map_state.get("all_drawings")
    if all_drawings:
        return all_drawings[-1]
    return None


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #


def main() -> None:
    from streamlit_folium import st_folium

    index, date, max_area = _sidebar()

    st.title("Interactive Earth-Observation Explorer")
    st.write(
        "Pull **live Sentinel-2 L2A** imagery and render a chosen spectral index "
        "on the map. The index maths is reused from the eo-monitor package."
    )

    fmap = _build_map(st.session_state.get("overlay"))

    run = st.button(
        "Load imagery & compute index",
        type="primary",
        disabled=not render.EO_MONITOR_AVAILABLE,
    )

    map_state = st_folium(
        fmap,
        height=600,
        width=None,
        returned_objects=["last_active_drawing", "all_drawings"],
        key="eo_map",
    )

    drawing = _latest_drawing(map_state)
    if drawing is not None:
        st.session_state["aoi_geojson"] = drawing

    if run:
        _handle_request(date, index, max_area)

    _show_result()

    with st.expander("About this app"):
        st.markdown(
            """
            - **Data:** Sentinel-2 L2A via [Earth Search](https://earth-search.aws.element84.com/v1) (no auth).
            - **Indices:** reused from the eo-monitor package in this repository.
            - **Caching:** STAC queries and loads are cached by AOI, date, and index.
            """
        )


def _handle_request(date: _dt.date, index: str, max_area: float) -> None:
    """Validate the saved AOI, fetch a scene, compute the index, save the overlay."""
    geojson = st.session_state.get("aoi_geojson")
    if not geojson:
        st.warning("Draw an area of interest on the map first.")
        return

    try:
        bbox = stac.aoi_bbox_from_geojson(geojson)
    except ValueError as exc:
        st.error(f"Could not read the drawn area: {exc}")
        return

    validation = stac.validate_aoi(bbox, max_area_km2=max_area)
    if not validation.ok:
        st.warning(validation.message)
        return

    with st.spinner("Searching Earth Search for the least-cloudy scene..."):
        try:
            meta, dataset = _cached_find_and_load(tuple(bbox), date.isoformat(), index)
        except Exception as exc:  # noqa: BLE001 - surface network/STAC errors nicely
            st.error(f"Failed to query or load imagery: {exc}")
            return

    if dataset is None:
        st.warning(
            "No suitable Sentinel-2 scene was found for that area and date. "
            "Try a different date or a larger area."
        )
        st.session_state.pop("overlay", None)
        return

    with st.spinner("Computing the index..."):
        try:
            overlay = render.build_index_overlay(dataset, index, meta=meta)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to compute or render the index: {exc}")
            return

    st.session_state["overlay"] = overlay
    st.rerun()


def _show_result() -> None:
    """Show the scene caption and index statistics for the saved overlay."""
    overlay = st.session_state.get("overlay")
    if not overlay:
        return

    meta = overlay.get("meta") or {}
    scene_id = meta.get("id")
    if scene_id:
        cloud = meta.get("cloud_cover")
        cloud_txt = f"{cloud:.1f}%" if isinstance(cloud, (int, float)) else "n/a"
        st.success(f"Scene `{scene_id}` from {meta.get('datetime')} (cloud cover {cloud_txt}).")

    stats = overlay.get("stats") or {}
    if stats:
        name = overlay["name"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{name} min", f"{stats['min']:.3f}")
        col2.metric(f"{name} mean", f"{stats['mean']:.3f}")
        col3.metric(f"{name} max", f"{stats['max']:.3f}")
        col4.metric("Valid pixels", f"{stats['valid_fraction'] * 100:.0f}%")

    geotiff = overlay.get("geotiff")
    if geotiff:
        st.download_button(
            "⬇ Download GeoTIFF",
            data=geotiff,
            file_name=overlay.get("geotiff_name", f"{overlay['name']}.tif"),
            mime="image/tiff",
            help=(
                "The computed index as a georeferenced GeoTIFF (native projected "
                "CRS, float32, NaN nodata) — drop it straight into QGIS or rasterio."
            ),
        )


if __name__ == "__main__":
    main()
