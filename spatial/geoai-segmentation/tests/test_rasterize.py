"""Tests for the pure file-pairing helpers in scripts/rasterize_spacenet.py.

Only ``extract_key`` and ``pair_files`` are exercised here; they use just ``re``
and ``pathlib``, so they run without rasterio or geopandas. The heavy
``rasterize_pair`` function needs the geospatial stack and is not tested here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rasterize_spacenet.py"
_spec = importlib.util.spec_from_file_location("rasterize_spacenet", _SCRIPT)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def test_extract_key_spacenet_names():
    assert rs.extract_key("RGB-PanSharpen_AOI_2_Vegas_img1") == "img1"
    assert rs.extract_key("buildings_AOI_2_Vegas_img1") == "img1"
    assert rs.extract_key("img42") == "img42"


def test_extract_key_falls_back_to_stem():
    # No img<N> token, so the whole stem is the key.
    assert rs.extract_key("scene_abc") == "scene_abc"


def test_extract_key_custom_regex_without_group():
    # A pattern with no capture group returns the whole match.
    assert rs.extract_key("tile_007", key_regex=r"\d+$") == "007"


def test_pair_files_matches_on_key():
    images = [
        "/img/RGB-PanSharpen_AOI_2_Vegas_img1.tif",
        "/img/RGB-PanSharpen_AOI_2_Vegas_img2.tif",
    ]
    labels = [
        "/lbl/buildings_AOI_2_Vegas_img1.geojson",
        "/lbl/buildings_AOI_2_Vegas_img2.geojson",
    ]
    pairs = rs.pair_files(images, labels)
    assert len(pairs) == 2
    assert {rs.extract_key(p[0].stem) for p in pairs} == {"img1", "img2"}
    # Each tuple is (image, label, stem).
    image, label, stem = pairs[0]
    assert image.suffix == ".tif"
    assert label.suffix == ".geojson"
    assert stem == image.stem


def test_pair_files_skips_images_without_a_label():
    images = ["/img/x_img1.tif", "/img/x_img9.tif"]
    labels = ["/lbl/y_img1.geojson"]
    pairs = rs.pair_files(images, labels)
    assert len(pairs) == 1
    assert pairs[0][2] == "x_img1"


def test_pair_files_orders_by_stem():
    images = ["/img/a_img2.tif", "/img/a_img1.tif"]
    labels = ["/lbl/b_img1.geojson", "/lbl/b_img2.geojson"]
    stems = [stem for _img, _lbl, stem in rs.pair_files(images, labels)]
    assert stems == sorted(stems)
