"""Smoke tests for the EO Explorer app.

These tests import the app modules and exercise the **pure** helper functions
that have no heavy third-party dependencies. They must pass with only the
standard library (plus numpy, which the colour helpers use) installed -- i.e.
without streamlit, folium, pystac-client, odc-stac, or even eo-monitor.

Heavy imports inside ``app.render`` / ``app.stac`` are guarded so that importing
the modules at test-collection time does not pull in optional dependencies.
"""

from __future__ import annotations

import math

import pytest

# Importing the modules must not require streamlit / folium / pystac etc.
from app import render, stac

# --------------------------------------------------------------------------- #
# aoi_bbox_from_geojson
# --------------------------------------------------------------------------- #

SIMPLE_POLYGON = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [10.0, 50.0],
                [10.5, 50.0],
                [10.5, 50.4],
                [10.0, 50.4],
                [10.0, 50.0],
            ]
        ],
    },
}


def test_aoi_bbox_from_feature():
    bbox = stac.aoi_bbox_from_geojson(SIMPLE_POLYGON)
    assert bbox == (10.0, 50.0, 10.5, 50.4)


def test_aoi_bbox_from_bare_geometry():
    bbox = stac.aoi_bbox_from_geojson(SIMPLE_POLYGON["geometry"])
    assert bbox == (10.0, 50.0, 10.5, 50.4)


def test_aoi_bbox_from_feature_collection():
    fc = {"type": "FeatureCollection", "features": [SIMPLE_POLYGON]}
    bbox = stac.aoi_bbox_from_geojson(fc)
    assert bbox == (10.0, 50.0, 10.5, 50.4)


def test_aoi_bbox_rejects_empty():
    with pytest.raises(ValueError):
        stac.aoi_bbox_from_geojson({"type": "FeatureCollection", "features": []})


def test_aoi_bbox_rejects_non_dict():
    with pytest.raises(ValueError):
        stac.aoi_bbox_from_geojson("not-a-dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# bbox_area_km2 / validate_aoi
# --------------------------------------------------------------------------- #


def test_bbox_area_is_positive_and_reasonable():
    # ~0.5 deg lon x 0.4 deg lat near 50N -> a few thousand km^2.
    area = stac.bbox_area_km2((10.0, 50.0, 10.5, 50.4))
    assert area > 0
    assert 1000 < area < 3000


def test_validate_aoi_accepts_small_area():
    bbox = (10.0, 50.0, 10.1, 50.1)  # small box
    result = stac.validate_aoi(bbox, max_area_km2=stac.DEFAULT_MAX_AREA_KM2)
    assert result.ok is True
    assert result.area_km2 > 0


def test_validate_aoi_rejects_oversized_area():
    bbox = (0.0, 0.0, 20.0, 20.0)  # enormous box -> hundreds of thousands km^2
    result = stac.validate_aoi(bbox, max_area_km2=stac.DEFAULT_MAX_AREA_KM2)
    assert result.ok is False
    assert "larger than" in result.message
    assert result.area_km2 > stac.DEFAULT_MAX_AREA_KM2


def test_validate_aoi_rejects_zero_area():
    bbox = (10.0, 50.0, 10.0, 50.0)  # a point
    result = stac.validate_aoi(bbox)
    assert result.ok is False


def test_validate_aoi_rejects_wrong_length():
    result = stac.validate_aoi((1.0, 2.0, 3.0))  # type: ignore[arg-type]
    assert result.ok is False


# --------------------------------------------------------------------------- #
# cache_key
# --------------------------------------------------------------------------- #


def test_cache_key_is_deterministic():
    bbox = (10.0, 50.0, 10.5, 50.4)
    k1 = stac.cache_key(bbox, "2024-06-15", "NDVI")
    k2 = stac.cache_key(bbox, "2024-06-15", "NDVI")
    assert k1 == k2
    assert k1.startswith("eo-explorer:")


def test_cache_key_index_is_case_insensitive():
    bbox = (10.0, 50.0, 10.5, 50.4)
    assert stac.cache_key(bbox, "2024-06-15", "ndvi") == stac.cache_key(bbox, "2024-06-15", "NDVI")


def test_cache_key_changes_with_inputs():
    bbox = (10.0, 50.0, 10.5, 50.4)
    base = stac.cache_key(bbox, "2024-06-15", "NDVI")
    assert base != stac.cache_key(bbox, "2024-06-16", "NDVI")
    assert base != stac.cache_key(bbox, "2024-06-15", "NDWI")
    assert base != stac.cache_key((10.0, 50.0, 10.5, 50.5), "2024-06-15", "NDVI")


def test_cache_key_rounds_bbox():
    a = stac.cache_key((10.0, 50.0, 10.5, 50.4), "2024-06-15", "NDVI")
    b = stac.cache_key((10.0000000001, 50.0, 10.5, 50.4), "2024-06-15", "NDVI")
    assert a == b


# --------------------------------------------------------------------------- #
# date_window
# --------------------------------------------------------------------------- #


def test_date_window_format():
    window = stac.date_window("2024-06-15", days=10)
    assert window == "2024-06-05/2024-06-25"


# --------------------------------------------------------------------------- #
# render: registry + colour helpers (numpy only)
# --------------------------------------------------------------------------- #


def test_index_registry_complete():
    names = render.list_indices()
    # The original three remain available...
    for required in ("NDVI", "NDWI", "NDMI"):
        assert required in names
    # ...and the catalogue has grown well beyond them.
    assert len(names) > 20
    valid_categories = {"Vegetation", "Water", "Soil", "Built-up", "Snow", "Fire"}
    for name in names:
        spec = render.INDEX_REGISTRY[name]
        assert spec.vmax > spec.vmin
        assert callable(spec.func)
        assert isinstance(spec.bands, tuple) and len(spec.bands) >= 2
        assert spec.category in valid_categories


def test_index_registry_bands_match_stac():
    """Every registry index must have a band entry in stac.INDEX_BANDS covering
    the same assets it computes from."""
    for name, spec in render.INDEX_REGISTRY.items():
        assert name in stac.INDEX_BANDS
        assert set(spec.bands) == set(stac.INDEX_BANDS[name])


def test_list_indices_by_category():
    cats = render.list_indices_by_category()
    assert set(cats) == {"Vegetation", "Water", "Soil", "Built-up", "Snow", "Fire"}
    # Each category lists at least one index, and the union is the full set.
    flat = [n for names in cats.values() for n in names]
    assert sorted(flat) == sorted(render.list_indices())


def test_normalize_clips_and_scales():
    np = pytest.importorskip("numpy")
    out = render.normalize(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), vmin=0.0, vmax=1.0)
    assert out.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_normalize_rejects_bad_range():
    with pytest.raises(ValueError):
        render.normalize([0.0, 1.0], vmin=1.0, vmax=1.0)


def test_index_stats_handles_all_nan():
    np = pytest.importorskip("numpy")
    arr = np.full((3, 3), np.nan)
    stats = render.index_stats(arr)
    assert stats["valid_fraction"] == 0.0
    assert math.isnan(stats["mean"])


def test_index_stats_basic():
    np = pytest.importorskip("numpy")
    arr = np.array([[0.0, 1.0], [np.nan, 0.5]])
    stats = render.index_stats(arr)
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0
    assert stats["valid_fraction"] == 0.75


def test_fallback_index_math_matches_formula():
    """Even without eo-monitor the fallback NDVI must equal (nir-red)/(nir+red)."""
    np = pytest.importorskip("numpy")
    red = np.array([0.1, 0.2])
    nir = np.array([0.5, 0.4])
    expected = (nir - red) / (nir + red)
    got = render._ndvi(nir, red)
    assert np.allclose(got, expected)


def test_fallback_ndwi_ndmi_match_formula():
    """The two other smoke-tested fallbacks match their normalised-difference."""
    np = pytest.importorskip("numpy")
    green = np.array([0.3, 0.4])
    nir = np.array([0.1, 0.6])
    swir = np.array([0.2, 0.5])
    assert np.allclose(render._ndwi(green, nir), (green - nir) / (green + nir))
    assert np.allclose(render._ndmi(nir, swir), (nir - swir) / (nir + swir))


def test_registry_funcs_compute_known_values():
    """Spot-check the smoke-tested fallback registry funcs on small arrays.

    These three (NDVI/NDWI/NDMI) carry local fallbacks so they compute even
    without eo-monitor; the wider catalogue resolves from eo-monitor in the
    production path and is covered by eo-monitor's own tests."""
    np = pytest.importorskip("numpy")
    # NDVI: (nir-red)/(nir+red) = (0.5-0.1)/(0.5+0.1) = 2/3.
    f_ndvi = render.INDEX_REGISTRY["NDVI"].func
    assert np.allclose(f_ndvi(np.array([0.5]), np.array([0.1])), np.array([2.0 / 3.0]))
    # NDMI: (nir-swir)/(nir+swir) = (0.6-0.2)/(0.6+0.2) = 0.5.
    f_ndmi = render.INDEX_REGISTRY["NDMI"].func
    assert np.allclose(f_ndmi(np.array([0.6]), np.array([0.2])), np.array([0.5]))


# --------------------------------------------------------------------------- #
# aoi_bbox_from_geojson: MultiPolygon + GeometryCollection
# --------------------------------------------------------------------------- #


def test_aoi_bbox_from_multipolygon():
    geom = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
            [[[5.0, 5.0], [6.0, 5.0], [6.0, 7.0], [5.0, 7.0], [5.0, 5.0]]],
        ],
    }
    bbox = stac.aoi_bbox_from_geojson(geom)
    assert bbox == (0.0, 0.0, 6.0, 7.0)


def test_aoi_bbox_from_geometry_collection():
    geom = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": [3.0, 4.0]},
            {
                "type": "Polygon",
                "coordinates": [[[-1.0, -1.0], [2.0, -1.0], [2.0, 2.0], [-1.0, 2.0], [-1.0, -1.0]]],
            },
        ],
    }
    bbox = stac.aoi_bbox_from_geojson(geom)
    assert bbox == (-1.0, -1.0, 3.0, 4.0)


def test_aoi_bbox_feature_collection_unions_features():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0], [10.0, 10.0]]
                    ],
                },
            },
        ],
    }
    bbox = stac.aoi_bbox_from_geojson(fc)
    assert bbox == (0.0, 0.0, 11.0, 11.0)


def test_aoi_bbox_ignores_z_coordinate():
    geom = {
        "type": "Polygon",
        "coordinates": [
            [[1.0, 2.0, 100.0], [3.0, 2.0, 100.0], [3.0, 4.0, 100.0], [1.0, 2.0, 100.0]]
        ],
    }
    bbox = stac.aoi_bbox_from_geojson(geom)
    assert bbox == (1.0, 2.0, 3.0, 4.0)


def test_aoi_bbox_rejects_missing_type():
    with pytest.raises(ValueError):
        stac.aoi_bbox_from_geojson({"coordinates": [[0.0, 0.0]]})


# --------------------------------------------------------------------------- #
# crosses_antimeridian
# --------------------------------------------------------------------------- #


def test_crosses_antimeridian_true_when_lons_out_of_order():
    assert stac.crosses_antimeridian((170.0, 10.0, -170.0, 20.0)) is True


def test_crosses_antimeridian_false_for_normal_bbox():
    assert stac.crosses_antimeridian((10.0, 50.0, 10.5, 50.4)) is False


def test_crosses_antimeridian_false_for_wrong_length():
    assert stac.crosses_antimeridian((1.0, 2.0, 3.0)) is False


# --------------------------------------------------------------------------- #
# validate_aoi: hardened edge cases
# --------------------------------------------------------------------------- #


def test_validate_aoi_rejects_none():
    result = stac.validate_aoi(None)  # type: ignore[arg-type]
    assert result.ok is False


def test_validate_aoi_rejects_antimeridian():
    result = stac.validate_aoi((170.0, 10.0, -170.0, 20.0))
    assert result.ok is False
    assert "date line" in result.message


def test_validate_aoi_rejects_inverted_latitude():
    # Longitudes in order so the antimeridian check passes, but north < south.
    result = stac.validate_aoi((10.0, 50.0, 10.5, 49.0))
    assert result.ok is False
    assert "inverted" in result.message


def test_validate_aoi_rejects_out_of_range_longitude():
    result = stac.validate_aoi((10.0, 50.0, 200.0, 50.4))
    assert result.ok is False
    assert "Longitude" in result.message


def test_validate_aoi_rejects_out_of_range_latitude():
    result = stac.validate_aoi((10.0, -100.0, 10.5, 50.4))
    assert result.ok is False
    assert "Latitude" in result.message


def test_validate_aoi_rejects_nan_coordinate():
    result = stac.validate_aoi((10.0, 50.0, float("nan"), 50.4))
    assert result.ok is False


def test_validate_aoi_rejects_non_numeric():
    result = stac.validate_aoi(("a", "b", "c", "d"))  # type: ignore[arg-type]
    assert result.ok is False


def test_validate_aoi_zero_width_line_is_rejected():
    # Same longitude on both edges: a vertical line, zero area.
    result = stac.validate_aoi((10.0, 50.0, 10.0, 50.4))
    assert result.ok is False
    assert "no extent" in result.message


def test_validate_aoi_respects_custom_max_area():
    bbox = (10.0, 50.0, 10.1, 50.1)  # ~80 km^2 near 50N
    big = stac.validate_aoi(bbox, max_area_km2=1000.0)
    small = stac.validate_aoi(bbox, max_area_km2=1.0)
    assert big.ok is True
    assert small.ok is False
    assert "larger than" in small.message


# --------------------------------------------------------------------------- #
# cache_key: determinism + collision resistance
# --------------------------------------------------------------------------- #


def test_cache_key_distinct_across_many_requests():
    """A grid of (bbox, date, index) inputs must yield all-distinct keys."""
    keys = set()
    requests = []
    for lon in (10.0, 11.0, 12.0):
        for day in ("2024-06-15", "2024-07-15"):
            for index in render.list_indices():
                bbox = (lon, 50.0, lon + 0.5, 50.4)
                requests.append((bbox, day, index))
                keys.add(stac.cache_key(bbox, day, index))
    assert len(keys) == len(requests)


def test_cache_key_swapped_fields_do_not_collide():
    # Date and index text must not be interchangeable in the digest.
    a = stac.cache_key((1.0, 2.0, 3.0, 4.0), "NDVI", "2024-06-15")
    b = stac.cache_key((1.0, 2.0, 3.0, 4.0), "2024-06-15", "NDVI")
    assert a != b


def test_cache_key_rounding_boundary_separates_close_aois():
    # A difference above the 6-dp rounding resolution must change the key.
    a = stac.cache_key((10.0, 50.0, 10.5, 50.4), "2024-06-15", "NDVI")
    b = stac.cache_key((10.00001, 50.0, 10.5, 50.4), "2024-06-15", "NDVI")
    assert a != b


def test_cache_key_format():
    key = stac.cache_key((1.0, 2.0, 3.0, 4.0), "2024-06-15", "NDVI")
    prefix, _, digest = key.partition(":")
    assert prefix == "eo-explorer"
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


# --------------------------------------------------------------------------- #
# date_window: edge cases
# --------------------------------------------------------------------------- #


def test_date_window_zero_days():
    assert stac.date_window("2024-06-15", days=0) == "2024-06-15/2024-06-15"


def test_date_window_crosses_month_boundary():
    assert stac.date_window("2024-03-03", days=5) == "2024-02-27/2024-03-08"


def test_date_window_crosses_year_boundary():
    assert stac.date_window("2024-01-02", days=5) == "2023-12-28/2024-01-07"


def test_date_window_leap_day():
    assert stac.date_window("2024-02-28", days=1) == "2024-02-27/2024-02-29"


def test_date_window_rejects_negative_days():
    with pytest.raises(ValueError):
        stac.date_window("2024-06-15", days=-1)


def test_date_window_rejects_bad_date():
    with pytest.raises(ValueError):
        stac.date_window("not-a-date")


# --------------------------------------------------------------------------- #
# render: colormap / normalisation (pure numpy + matplotlib)
# --------------------------------------------------------------------------- #


def test_normalize_preserves_nan():
    np = pytest.importorskip("numpy")
    out = render.normalize(np.array([np.nan, 0.5]), vmin=0.0, vmax=1.0)
    assert math.isnan(float(out[0]))
    assert out[1] == 0.5


def test_normalize_midpoint():
    np = pytest.importorskip("numpy")
    out = render.normalize(np.array([0.0, 1.0, 2.0]), vmin=0.0, vmax=2.0)
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_colorize_shape_and_dtype():
    np = pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    arr = np.array([[-0.2, 0.4], [0.9, 0.0]])
    rgba = render.colorize(arr, vmin=-0.2, vmax=0.9, colormap="RdYlGn")
    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.dtype("uint8")


def test_colorize_makes_nan_transparent():
    np = pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    arr = np.array([[np.nan, 0.5]])
    rgba = render.colorize(arr, vmin=0.0, vmax=1.0, colormap="viridis")
    assert rgba[0, 0, 3] == 0  # NaN -> transparent
    assert rgba[0, 1, 3] == 255  # finite -> opaque


def test_colorize_endpoints_hit_colormap_extremes():
    np = pytest.importorskip("numpy")
    matplotlib = pytest.importorskip("matplotlib")
    cmap = matplotlib.colormaps["viridis"]
    # colorize multiplies by 255 then casts to uint8, which truncates; mirror
    # that here rather than rounding so the comparison reflects production maths.
    low = tuple(int(c * 255) for c in cmap(0.0))
    high = tuple(int(c * 255) for c in cmap(1.0))
    arr = np.array([[0.0, 1.0]])
    rgba = render.colorize(arr, vmin=0.0, vmax=1.0, colormap="viridis")
    assert tuple(int(v) for v in rgba[0, 0]) == low
    assert tuple(int(v) for v in rgba[0, 1]) == high


def test_colorize_clips_out_of_range_values():
    np = pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    # Values beyond [vmin, vmax] must clamp to the same colour as the endpoints.
    arr = np.array([[-5.0, 0.0]])  # -5 clamps to vmin (0.0)
    rgba = render.colorize(arr, vmin=0.0, vmax=1.0, colormap="viridis")
    assert tuple(int(v) for v in rgba[0, 0]) == tuple(int(v) for v in rgba[0, 1])


# --------------------------------------------------------------------------- #
# stac geometry helpers: bbox_center / bbox_aspect_ratio / suggest_zoom
# --------------------------------------------------------------------------- #


def test_bbox_center_known_box():
    # Midpoint of (0,0,2,1) is the arithmetic mean of each axis.
    assert stac.bbox_center((0.0, 0.0, 2.0, 1.0)) == (1.0, 0.5)


def test_bbox_center_negative_box():
    assert stac.bbox_center((-10.0, -4.0, -6.0, 2.0)) == (-8.0, -1.0)


def test_bbox_aspect_ratio_at_equator():
    # At the equator cos(lat)~1, so a 2deg x 1deg box has aspect ~2.0.
    ratio = stac.bbox_aspect_ratio((0.0, 0.0, 2.0, 1.0))
    assert ratio == pytest.approx(2.0, rel=1e-3)


def test_bbox_aspect_ratio_shrinks_with_latitude():
    # The same degree box is "narrower" in km the further north it sits, so the
    # cosine correction makes the width/height ratio fall.
    equator = stac.bbox_aspect_ratio((0.0, 0.0, 2.0, 1.0))
    north = stac.bbox_aspect_ratio((0.0, 60.0, 2.0, 61.0))
    assert north < equator
    # cos(60.5deg) ~ 0.4924, times the 2.0 equator ratio.
    assert north == pytest.approx(2.0 * math.cos(math.radians(60.5)), rel=1e-3)


def test_bbox_aspect_ratio_rejects_zero_height():
    with pytest.raises(ValueError):
        stac.bbox_aspect_ratio((0.0, 5.0, 2.0, 5.0))


def test_suggest_zoom_known_values():
    # zoom = floor(log2(360 / dlon)).
    assert stac.suggest_zoom((-180.0, 0.0, 180.0, 0.0)) == 0  # 360deg -> log2(1)=0
    assert stac.suggest_zoom((0.0, 0.0, 180.0, 0.0)) == 1  # 180deg -> log2(2)=1
    assert stac.suggest_zoom((0.0, 0.0, 90.0, 0.0)) == 2  # 90deg -> log2(4)=2


def test_suggest_zoom_is_monotonic():
    # A smaller bbox must zoom in at least as far as a larger one.
    big = stac.suggest_zoom((0.0, 50.0, 20.0, 50.1))
    medium = stac.suggest_zoom((0.0, 50.0, 1.0, 50.1))
    small = stac.suggest_zoom((0.0, 50.0, 0.1, 50.1))
    assert big <= medium <= small


def test_suggest_zoom_zero_width_returns_max():
    assert stac.suggest_zoom((10.0, 50.0, 10.0, 50.1)) == 22


# --------------------------------------------------------------------------- #
# render numpy helpers: percentile_stretch / histogram / downsample
# --------------------------------------------------------------------------- #


def test_percentile_stretch_known_vector():
    np = pytest.importorskip("numpy")
    # On 0..100 the 2nd and 98th percentiles are 2.0 and 98.0 by linear interp.
    vmin, vmax = render.percentile_stretch(np.arange(0, 101, dtype="float64"))
    assert vmin == pytest.approx(2.0)
    assert vmax == pytest.approx(98.0)


def test_percentile_stretch_ignores_nan():
    np = pytest.importorskip("numpy")
    arr = np.array([np.nan, 0.0, 50.0, 100.0, np.nan])
    vmin, vmax = render.percentile_stretch(arr, lo=0, hi=100)
    assert vmin == pytest.approx(0.0)
    assert vmax == pytest.approx(100.0)


def test_percentile_stretch_all_nan_returns_nan():
    np = pytest.importorskip("numpy")
    vmin, vmax = render.percentile_stretch(np.full(5, np.nan))
    assert math.isnan(vmin)
    assert math.isnan(vmax)


def test_percentile_stretch_rejects_bad_range():
    with pytest.raises(ValueError):
        render.percentile_stretch([0.0, 1.0], lo=98, hi=2)


def test_histogram_known_counts():
    np = pytest.importorskip("numpy")
    # Edges [0, 1.5, 3.5]: {0,1} fall in the first bin, {2,3} in the second.
    counts, edges = render.histogram(np.array([0.0, 1.0, 2.0, 3.0, np.nan]), bins=[0, 1.5, 3.5])
    assert counts.tolist() == [2, 2]
    assert edges.tolist() == [0.0, 1.5, 3.5]


def test_histogram_all_nan_is_empty():
    np = pytest.importorskip("numpy")
    counts, _edges = render.histogram(np.full(4, np.nan), bins=4)
    assert counts.sum() == 0


def test_downsample_shape_and_values():
    np = pytest.importorskip("numpy")
    arr = np.arange(100).reshape(10, 10)
    out = render.downsample(arr, max_dim=5)
    assert max(out.shape) <= 5
    # step = ceil(10/5) = 2, so out[0,0] is arr[0,0] and out[1,0] is arr[2,0].
    assert out[0, 0] == 0
    assert out[1, 0] == 20


def test_downsample_leaves_small_array_unchanged():
    np = pytest.importorskip("numpy")
    arr = np.arange(9).reshape(3, 3)
    out = render.downsample(arr, max_dim=5)
    assert out.shape == (3, 3)
    assert np.array_equal(out, arr)


def test_downsample_single_pixel():
    np = pytest.importorskip("numpy")
    arr = np.array([[42.0]])
    out = render.downsample(arr, max_dim=1)
    assert out.shape == (1, 1)
    assert out[0, 0] == 42.0


def test_downsample_rejects_bad_args():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        render.downsample(np.zeros((4, 4)), max_dim=0)
    with pytest.raises(ValueError):
        render.downsample(np.zeros(4), max_dim=2)  # 1-D not allowed
