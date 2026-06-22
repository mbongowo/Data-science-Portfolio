"""Known-answer tests for pure-numpy tiling. Requires only numpy."""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.tiling import stitch, tile_indices


def test_tile_indices_exact_grid_no_overlap():
    # 4x4 image, tile 2, overlap 0 -> a clean 2x2 grid of 2x2 windows.
    wins = tile_indices(4, 4, tile=2, overlap=0)
    assert wins == [
        (0, 2, 0, 2),
        (0, 2, 2, 4),
        (2, 4, 0, 2),
        (2, 4, 2, 4),
    ]


def test_tile_indices_covers_every_pixel():
    # Non-divisible size: 5x5 with tile 2 must still cover all 25 pixels.
    h, w, tile = 5, 5, 2
    wins = tile_indices(h, w, tile=tile, overlap=0)
    covered = np.zeros((h, w), dtype=bool)
    for r0, r1, c0, c1 in wins:
        covered[r0:r1, c0:c1] = True
    assert covered.all()


def test_tile_indices_snaps_last_window_in_bounds():
    # 5 wide, tile 2, stride 2: starts 0,2 then snap to 3 so window ends at 5.
    wins = tile_indices(2, 5, tile=2, overlap=0)
    cols = sorted({(c0, c1) for _, _, c0, c1 in wins})
    assert cols == [(0, 2), (2, 4), (3, 5)]
    # nothing runs out of bounds
    assert all(c1 <= 5 for _, _, _, c1 in wins)


def test_tile_indices_overlap_stride():
    # tile 4, overlap 2 -> stride 2.
    wins = tile_indices(2, 8, tile=4, overlap=2)
    cols = [(c0, c1) for _, _, c0, c1 in wins]
    assert cols == [(0, 4), (2, 6), (4, 8)]


def test_tile_indices_tile_larger_than_image():
    # tile bigger than the image -> a single full-extent window.
    wins = tile_indices(3, 5, tile=10, overlap=0)
    assert wins == [(0, 3, 0, 5)]


def test_tile_indices_invalid_args():
    with pytest.raises(ValueError):
        tile_indices(0, 4, tile=2)
    with pytest.raises(ValueError):
        tile_indices(4, 4, tile=0)
    with pytest.raises(ValueError):
        tile_indices(4, 4, tile=2, overlap=2)  # overlap == tile
    with pytest.raises(ValueError):
        tile_indices(4, 4, tile=2, overlap=-1)


def test_stitch_round_trip_no_overlap_2d():
    # overlap=0 must reconstruct the original array exactly.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 9, size=(6, 6)).astype(np.float64)
    wins = tile_indices(6, 6, tile=3, overlap=0)
    tiles = [img[r0:r1, c0:c1] for r0, r1, c0, c1 in wins]
    out = stitch(tiles, wins, 6, 6)
    assert np.array_equal(out, img)


def test_stitch_round_trip_with_overlap_averages_to_original():
    # Even with overlap, averaging identical source pixels recovers the image.
    rng = np.random.default_rng(1)
    img = rng.random((5, 7))
    wins = tile_indices(5, 7, tile=3, overlap=1)
    tiles = [img[r0:r1, c0:c1] for r0, r1, c0, c1 in wins]
    out = stitch(tiles, wins, 5, 7)
    assert np.allclose(out, img)


def test_stitch_overlap_is_mean_of_tiles():
    # Two windows overlap on the middle column; stitch averages them.
    # image is 1x4; windows (0,1,0,2) and (0,1,2,4) don't overlap, so build
    # an explicit overlapping pair instead.
    positions = [(0, 1, 0, 3), (0, 1, 1, 4)]
    left = np.array([[10.0, 20.0, 30.0]])  # covers cols 0,1,2
    right = np.array([[40.0, 50.0, 60.0]])  # covers cols 1,2,3
    out = stitch([left, right], positions, 1, 4)
    # col0: 10 ; col1: mean(20,40)=30 ; col2: mean(30,50)=40 ; col3: 60
    assert np.allclose(out, [[10.0, 30.0, 40.0, 60.0]])


def test_stitch_round_trip_3d_channels():
    rng = np.random.default_rng(2)
    img = rng.random((4, 4, 3))
    wins = tile_indices(4, 4, tile=2, overlap=0)
    tiles = [img[r0:r1, c0:c1] for r0, r1, c0, c1 in wins]
    out = stitch(tiles, wins, 4, 4)
    assert out.shape == (4, 4, 3)
    assert np.allclose(out, img)


def test_stitch_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        stitch([np.zeros((2, 2))], [(0, 2, 0, 2), (0, 2, 2, 4)], 2, 4)


def test_stitch_empty_raises():
    with pytest.raises(ValueError):
        stitch([], [], 2, 2)


def test_stitch_bad_tile_shape_raises():
    with pytest.raises(ValueError):
        stitch([np.zeros((3, 3))], [(0, 2, 0, 2)], 2, 2)


def test_stitch_incomplete_coverage_raises():
    # A single small window cannot cover the whole image.
    with pytest.raises(ValueError):
        stitch([np.zeros((2, 2))], [(0, 2, 0, 2)], 4, 4)
