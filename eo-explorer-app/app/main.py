"""Streamlit entry point for the EO Explorer app.

Run locally with::

    streamlit run app/main.py

Workflow
--------
1. The user draws an AOI on the leafmap map (draw control).
2. The sidebar offers a date picker and an index selector (NDVI / NDWI / NDMI).
3. On request the app finds the least-cloudy Sentinel-2 L2A scene near the date,
   loads the needed bands, computes the index using the reused ``eo-monitor``
   functions, and renders it as a coloured layer with a legend.

All STAC queries / loads are cached with ``st.cache_data`` keyed by the
AOI bbox + date + index so repeated requests are fast.
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
    """Find the best scene and load its bands, cached by (bbox, date, index).

    The explicit ``cache_key`` is folded into the arguments so the cache entry is
    deterministic and shareable across reruns.
    """
    key = stac.cache_key(bbox, date, index)  # noqa: F841 - documents the cache identity
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

    index = st.sidebar.selectbox(
        "Spectral index",
        options=render.list_indices(),
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
            "eo-monitor is not installed - index computation is disabled. "
            "Install it with `pip install -e ../eo-monitor`."
        )

    return index, date, float(max_area)


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #


def _get_drawn_geojson(map_obj):
    """Return the last-drawn GeoJSON feature from a leafmap map, or ``None``."""
    feat = getattr(map_obj, "user_roi", None)
    if feat:
        return feat
    drawn = getattr(map_obj, "draw_features", None)
    if drawn:
        return {"type": "FeatureCollection", "features": drawn}
    return None


def main() -> None:
    import leafmap.foliumap as leafmap

    index, date, max_area = _sidebar()

    st.title("Interactive Earth-Observation Explorer")
    st.write(
        "A shippable demo that pulls **live Sentinel-2 L2A** imagery and renders "
        "a chosen spectral index on the map. The index maths is reused from the "
        "flagship **eo-monitor** package."
    )

    m = leafmap.Map(center=[0, 20], zoom=3, draw_export=False)
    m.add_basemap("Esri.WorldImagery")

    geojson = _get_drawn_geojson(m)

    run = st.button("Load imagery & compute index", type="primary")

    if run:
        if not geojson:
            st.warning("Please draw an area of interest on the map first.")
        else:
            _handle_request(m, geojson, date, index, max_area)

    m.to_streamlit(height=620)

    with st.expander("About this app"):
        st.markdown(
            """
            * **Data:** Sentinel-2 L2A via [Earth Search](https://earth-search.aws.element84.com/v1) (no auth).
            * **Indices:** reused from [`eo-monitor`](https://github.com/JosephMbuh/eo-monitor).
            * **Caching:** STAC queries/loads are cached by AOI + date + index.
            """
        )


def _handle_request(m, geojson, date, index, max_area) -> None:
    """Validate the AOI, fetch the scene, compute and render the index."""
    try:
        bbox = stac.aoi_bbox_from_geojson(geojson)
    except ValueError as exc:
        st.error(f"Could not read the drawn area: {exc}")
        return

    validation = stac.validate_aoi(bbox, max_area_km2=max_area)
    if not validation.ok:
        st.warning(validation.message)
        return
    st.info(validation.message)

    if not render.EO_MONITOR_AVAILABLE:
        st.error(
            "Cannot compute the index because eo-monitor is not installed. "
            "Install it with `pip install -e ../eo-monitor` and rerun."
        )
        return

    with st.spinner("Searching Earth Search for the least-cloudy scene..."):
        try:
            meta, dataset = _cached_find_and_load(
                tuple(bbox), date.isoformat(), index
            )
        except Exception as exc:  # noqa: BLE001 - surface network/STAC errors nicely
            st.error(f"Failed to query or load imagery: {exc}")
            return

    if dataset is None:
        st.warning(
            "No suitable Sentinel-2 scene was found for that area and date. "
            "Try a different date or a larger date window."
        )
        return

    st.success(
        f"Scene `{meta['id']}` from {meta['datetime']} "
        f"(cloud cover {meta['cloud_cover']:.1f}%)."
    )

    try:
        data = render.add_index_layer(m, dataset, index)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to compute or render the index: {exc}")
        return

    stats = render.index_stats(data)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{index} min", f"{stats['min']:.3f}")
    col2.metric(f"{index} mean", f"{stats['mean']:.3f}")
    col3.metric(f"{index} max", f"{stats['max']:.3f}")
    col4.metric("Valid pixels", f"{stats['valid_fraction'] * 100:.0f}%")


if __name__ == "__main__":
    main()
