"""Tests for the pure-python datamodule helpers.

Tiling / splitting / normalisation are numpy-only and always run. The torch
``Dataset.__getitem__`` path is skipped gracefully when torch is missing.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from geoseg.datamodule import (
    TileSpec,
    compute_tile_grid,
    deterministic_split,
    normalize_per_band,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_tile_grid_exact_cover():
    tiles = compute_tile_grid(512, 512, tile_size=256)
    # 512/256 = 2 in each dim -> 4 non-overlapping tiles
    assert len(tiles) == 4
    for t in tiles:
        assert t.height == 256 and t.width == 256
        assert t.row + t.height <= 512
        assert t.col + t.width <= 512


def test_tile_grid_drops_partial():
    # 300x300 with 256 tiles, drop_partial -> only the top-left tile fits
    tiles = compute_tile_grid(300, 300, tile_size=256, drop_partial=True)
    assert len(tiles) == 1
    assert tiles[0].row == 0 and tiles[0].col == 0


def test_tile_grid_snaps_partial_when_kept():
    tiles = compute_tile_grid(300, 300, tile_size=256, drop_partial=False)
    # all tiles must remain in-bounds after snapping
    for t in tiles:
        assert t.row + t.height <= 300
        assert t.col + t.width <= 300


def test_tile_grid_overlap_stride():
    tiles = compute_tile_grid(512, 256, tile_size=256, stride=128)
    # rows at 0,128,256 -> 128 partial dropped at row 384; cols only 0
    rows = sorted({t.row for t in tiles})
    assert rows == [0, 128, 256]


def test_tile_grid_rectangular_exact_cover():
    # 200x300 with 100 tiles -> 2 rows x 3 cols = 6 tiles, no overlap
    tiles = compute_tile_grid(200, 300, tile_size=100)
    assert len(tiles) == 6
    assert sorted({t.row for t in tiles}) == [0, 100]
    assert sorted({t.col for t in tiles}) == [0, 100, 200]


def test_tile_grid_single_exact_fit():
    tiles = compute_tile_grid(100, 100, tile_size=100)
    assert len(tiles) == 1
    assert tiles[0] == TileSpec(0, 0, 100, 100)


def test_tile_grid_smaller_than_tile_drop():
    # raster smaller than one tile, dropping partials -> nothing fits
    assert compute_tile_grid(50, 50, tile_size=100, drop_partial=True) == []


def test_tile_grid_smaller_than_tile_keep_clamps_origin():
    # when partials are kept, the origin clamps to 0 (cannot go negative)
    tiles = compute_tile_grid(50, 50, tile_size=100, drop_partial=False)
    assert len(tiles) == 1
    assert tiles[0].row == 0 and tiles[0].col == 0


def test_tile_grid_snapping_deduplicates():
    # 300x300 with 256 tiles, keep partials: origins 0 and 44 in each axis
    tiles = compute_tile_grid(300, 300, tile_size=256, drop_partial=False)
    assert len(tiles) == 4  # 2x2 unique after snap-dedup
    assert sorted({t.row for t in tiles}) == [0, 44]
    keys = [t.key for t in tiles]
    assert len(keys) == len(set(keys))  # no duplicate windows


def test_tile_grid_overlap_all_in_bounds():
    tiles = compute_tile_grid(512, 512, tile_size=256, stride=128)
    assert len(tiles) == 9  # rows/cols at 0,128,256 each (384 partial dropped)
    for t in tiles:
        assert t.row + t.height <= 512
        assert t.col + t.width <= 512


def test_tile_grid_bad_tile_size_raises():
    with pytest.raises(ValueError):
        compute_tile_grid(256, 256, tile_size=0)


def test_tile_grid_bad_stride_raises():
    with pytest.raises(ValueError):
        compute_tile_grid(256, 256, tile_size=128, stride=0)


def test_deterministic_split_is_stable():
    keys = [f"tile_{i}" for i in range(200)]
    s1 = deterministic_split(keys, seed=42)
    s2 = deterministic_split(keys, seed=42)
    assert s1 == s2  # bit-for-bit reproducible


def test_deterministic_split_partitions_all_keys():
    keys = [f"tile_{i}" for i in range(200)]
    s = deterministic_split(keys, fractions=(0.7, 0.15, 0.15), seed=7)
    total = len(s["train"]) + len(s["val"]) + len(s["test"])
    assert total == len(keys)
    # no key appears in two splits
    assert set(s["train"]).isdisjoint(s["val"])
    assert set(s["train"]).isdisjoint(s["test"])
    assert set(s["val"]).isdisjoint(s["test"])


def test_deterministic_split_roughly_respects_fractions():
    keys = [f"tile_{i}" for i in range(2000)]
    s = deterministic_split(keys, fractions=(0.7, 0.15, 0.15), seed=1)
    assert abs(len(s["train"]) / 2000 - 0.7) < 0.05


def test_deterministic_split_seed_changes_assignment():
    keys = [f"tile_{i}" for i in range(200)]
    a = deterministic_split(keys, seed=1)
    b = deterministic_split(keys, seed=2)
    assert a != b


def test_deterministic_split_preserves_input_order():
    keys = [f"tile_{i}" for i in range(200)]
    s = deterministic_split(keys, seed=3)
    for name in ("train", "val", "test"):
        members = set(s[name])
        assert s[name] == [k for k in keys if k in members]


def test_deterministic_split_larger_fraction_gives_more_keys():
    keys = [f"tile_{i}" for i in range(2000)]
    small = deterministic_split(keys, fractions=(0.5, 0.25, 0.25), seed=5)
    large = deterministic_split(keys, fractions=(0.9, 0.05, 0.05), seed=5)
    assert len(large["train"]) > len(small["train"])
    assert abs(len(large["train"]) / 2000 - 0.9) < 0.03


def test_deterministic_split_unnormalised_fractions():
    # fractions need not sum to 1; they are renormalised internally
    keys = [f"tile_{i}" for i in range(2000)]
    s = deterministic_split(keys, fractions=(7, 1.5, 1.5), seed=9)
    assert abs(len(s["train"]) / 2000 - 0.7) < 0.05


def test_deterministic_split_bad_fractions_raise():
    with pytest.raises(ValueError):
        deterministic_split(["a"], fractions=(0.5, 0.5))  # not a 3-tuple
    with pytest.raises(ValueError):
        deterministic_split(["a"], fractions=(0.0, 0.0, 0.0))  # non-positive sum


def test_normalize_per_band_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    img = rng.normal(loc=[10, 20, 30], scale=[2, 4, 6], size=(64, 64, 3))
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (C,H,W)
    means = img.mean(axis=(1, 2))
    stds = img.std(axis=(1, 2))
    out = normalize_per_band(img, means, stds)
    assert out.shape == img.shape
    np.testing.assert_allclose(out.mean(axis=(1, 2)), 0.0, atol=1e-4)
    np.testing.assert_allclose(out.std(axis=(1, 2)), 1.0, atol=1e-2)


def test_normalize_per_band_band_mismatch_raises():
    img = np.zeros((3, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_per_band(img, means=[0.0, 0.0], stds=[1.0, 1.0])


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_dataset_item_shapes_synthetic(tmp_path):
    """Shape/alignment check on the torch path when torch is available."""
    import torch  # noqa: F401

    from geoseg.datamodule import GeoSegDataset, TileSpec

    class _StubDataset(GeoSegDataset):
        def _read_window(self, path, tile):  # bypass rasterio with synthetic data
            if "mask" in path:
                return (np.random.rand(1, tile.height, tile.width) > 0.5).astype(
                    np.float32
                )
            return np.random.rand(3, tile.height, tile.width).astype(np.float32)

    ds = _StubDataset(
        samples=[{"image": "img.tif", "mask": "mask.tif", "key": "a"}],
        tiles=[TileSpec(0, 0, 32, 32)],
        band_means=[0.0, 0.0, 0.0],
        band_stds=[1.0, 1.0, 1.0],
    )
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
    assert item["image"].shape[1:] == item["mask"].shape[1:]
