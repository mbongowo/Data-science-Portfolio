"""Hand-derived known-answer tests for flood-extent change and hectares.

Hand-laid pre/post boolean masks give exactly known flooded / permanent /
receded masks; a known flooded mask gives known hectares at a known pixel size; a
shape mismatch raises. Numpy + stdlib only.
"""

from __future__ import annotations

import numpy as np
import pytest

from floodmap.change import flood_extent, flood_stats


def test_flood_extent_masks_on_known_arrays() -> None:
    """flooded = post & ~pre, permanent = pre & post, receded = pre & ~post.

    pre  = [[1, 1, 0, 0]]
    post = [[1, 0, 1, 0]]
      col0 water both       -> permanent
      col1 water pre only    -> receded
      col2 water post only   -> flooded
      col3 dry both          -> nothing
    """
    pre = np.array([[True, True, False, False]])
    post = np.array([[True, False, True, False]])
    masks = flood_extent(pre, post)
    assert list(masks["flooded"][0]) == [False, False, True, False]
    assert list(masks["permanent_water"][0]) == [True, False, False, False]
    assert list(masks["receded"][0]) == [False, True, False, False]


def test_flood_stats_hectares_from_known_masks() -> None:
    """Hectares from known masks at 10 m pixels (each pixel = 100 m2 = 0.01 ha).

    On an 8-pixel scene: 3 flooded, 2 permanent, 1 receded ->
      flooded   0.03 ha, permanent 0.02 ha, receded 0.01 ha,
      flooded_fraction_of_scene = 3 / 8 = 0.375.
    """
    flooded = np.array([[True, True, True, False, False, False, False, False]])
    permanent = np.array([[False, False, False, True, True, False, False, False]])
    receded = np.array([[False, False, False, False, False, True, False, False]])
    masks = {
        "flooded": flooded,
        "permanent_water": permanent,
        "receded": receded,
    }
    stats = flood_stats(masks, pixel_size_m=10.0)
    assert stats["flooded_pixels"] == 3
    assert stats["flooded_hectares"] == pytest.approx(0.03)
    assert stats["permanent_water_hectares"] == pytest.approx(0.02)
    assert stats["receded_hectares"] == pytest.approx(0.01)
    assert stats["flooded_fraction_of_scene"] == pytest.approx(0.375)


def test_flood_stats_hectares_at_20m_pixels() -> None:
    """At 20 m pixels each pixel is 400 m2 = 0.04 ha."""
    masks = {
        "flooded": np.array([[True, True]]),
        "permanent_water": np.array([[False, False]]),
        "receded": np.array([[False, False]]),
    }
    stats = flood_stats(masks, pixel_size_m=20.0)
    assert stats["flooded_hectares"] == pytest.approx(0.08)


def test_invalid_inputs_raise() -> None:
    """Shape mismatch, missing mask, and bad pixel size raise ValueError."""
    with pytest.raises(ValueError):
        flood_extent(np.zeros((2, 2), bool), np.zeros((3, 3), bool))
    with pytest.raises(ValueError):
        flood_stats({"flooded": np.zeros((2, 2), bool)}, pixel_size_m=10.0)
    good = flood_extent(np.zeros((2, 2), bool), np.zeros((2, 2), bool))
    with pytest.raises(ValueError):
        flood_stats(good, pixel_size_m=0.0)
