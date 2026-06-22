"""Check the geometry-free facility parsing and cleaning rules.

Pure Python; no geopandas/osmnx dependency required.
"""

from __future__ import annotations

from access.facilities import clean_facility_records, geojson_features_to_records


def _feature(lon, lat, geom_type="Point", **props):
    return {
        "type": "Feature",
        "geometry": {"type": geom_type, "coordinates": [lon, lat]},
        "properties": props,
    }


def test_geojson_features_to_records_extracts_coords_and_props() -> None:
    feats = [_feature(11.5, 3.8, name="Yaounde Clinic", amenity="clinic")]
    recs = geojson_features_to_records(feats)
    assert len(recs) == 1
    assert recs[0]["lon"] == 11.5
    assert recs[0]["lat"] == 3.8
    assert recs[0]["name"] == "Yaounde Clinic"
    assert recs[0]["amenity"] == "clinic"


def test_geojson_skips_non_points_and_malformed() -> None:
    feats = [
        _feature(11.5, 3.8),
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1.0]}},  # too short
        {"type": "Feature"},  # no geometry
        "not a dict",
    ]
    recs = geojson_features_to_records(feats)
    assert len(recs) == 1
    assert recs[0]["lon"] == 11.5


def test_clean_drops_out_of_range_and_null_island() -> None:
    raw = [
        {"lon": 11.5, "lat": 3.8},  # valid
        {"lon": 200.0, "lat": 3.8},  # lon out of range
        {"lon": 11.5, "lat": 95.0},  # lat out of range
        {"lon": 0.0, "lat": 0.0},  # null island
        {"lon": None, "lat": 3.8},  # missing
        {"lon": "abc", "lat": 3.8},  # non-numeric
    ]
    cleaned = clean_facility_records(raw)
    assert len(cleaned) == 1
    assert cleaned[0]["lon"] == 11.5
    assert cleaned[0]["lat"] == 3.8


def test_clean_coerces_numeric_strings() -> None:
    cleaned = clean_facility_records([{"lon": "11.5", "lat": "3.8"}])
    assert cleaned == [{"lon": 11.5, "lat": 3.8}]


def test_parse_then_clean_pipeline() -> None:
    # The two pure steps compose: GeoJSON -> records -> cleaned records.
    feats = [
        _feature(11.5, 3.8, name="A"),
        _feature(0.0, 0.0, name="null-island"),
        _feature(999.0, 3.8, name="bad-lon"),
    ]
    cleaned = clean_facility_records(geojson_features_to_records(feats))
    assert [r["name"] for r in cleaned] == ["A"]
