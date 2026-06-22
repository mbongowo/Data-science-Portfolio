"""Hand-derived known-answer tests for the pure-numpy mask analytics core.

Every expected value below is worked out by hand on a tiny mask so a green test
proves the function is correct, not merely that it runs. None of these tests
import rasterio, shapely, geopandas, samgeo, or torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from samgeo_post.analytics import (
    area_hectares,
    count_objects,
    filter_by_area,
    label_components,
    mask_iou,
    pixels_to_area,
    region_props,
)


def test_two_separated_squares_give_two_components() -> None:
    """Two disjoint 2x2 squares -> 2 components, each of area 4.

    Layout (``.`` = background) on a 2x5 grid::

        X X . X X
        X X . X X

    The blank middle column separates them, so 4-connectivity finds two
    components, each four pixels.
    """
    mask = np.array(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
        ]
    )
    labeled = label_components(mask, connectivity=4)
    assert count_objects(labeled) == 2
    areas = sorted(p["area_px"] for p in region_props(labeled))
    assert areas == [4, 4]


def test_diagonal_pair_4_vs_8_connectivity() -> None:
    """A diagonal-touching pair is 2 components under 4-conn, 1 under 8-conn.

    Layout::

        X .
        . X

    Under 4-connectivity the two pixels share no edge, so there are two
    single-pixel components. Under 8-connectivity the diagonal link joins them
    into one component of area 2.
    """
    mask = np.array([[1, 0], [0, 1]])
    assert count_objects(label_components(mask, connectivity=4)) == 2
    assert count_objects(label_components(mask, connectivity=8)) == 1


def test_region_props_bbox_and_centroid_on_known_rectangle() -> None:
    """A solid rectangle has an exact bbox and centroid.

    A 3-row x 4-col block of ones placed at rows 1..3, cols 2..5 of a larger
    grid. Area = 12. bbox = (1, 2, 3, 5). The centroid is the mean pixel
    coordinate: rows 1,2,3 -> mean 2.0; cols 2,3,4,5 -> mean 3.5.
    """
    mask = np.zeros((6, 7), dtype=int)
    mask[1:4, 2:6] = 1
    props = region_props(label_components(mask, connectivity=4))
    assert len(props) == 1
    p = props[0]
    assert p["area_px"] == 12
    assert p["bbox"] == (1, 2, 3, 5)
    assert p["centroid"] == pytest.approx((2.0, 3.5))


def test_filter_by_area_drops_speckle_and_oversized_blob() -> None:
    """Area filtering drops a 1-px speckle and a huge blob, keeps the middle.

    Three objects with areas 1, 9, and 100. With ``min_px=4, max_px=50`` only
    the 9-pixel object survives.
    """
    mask = np.zeros((20, 20), dtype=int)
    mask[0, 0] = 1  # 1-px speckle
    mask[2:5, 2:5] = 1  # 3x3 = 9 px
    mask[8:18, 8:18] = 1  # 10x10 = 100 px
    props = region_props(label_components(mask, connectivity=4))
    assert sorted(p["area_px"] for p in props) == [1, 9, 100]

    kept = filter_by_area(props, min_px=4, max_px=50)
    assert len(kept) == 1
    assert kept[0]["area_px"] == 9


def test_pixels_to_area_and_hectares_exact() -> None:
    """Pixel->m2->ha conversions are exact.

    100 pixels at 0.5 m/px: each pixel is 0.25 m2, so 100 * 0.25 = 25 m2, which
    is 25 / 10000 = 0.0025 ha.
    """
    m2 = pixels_to_area(100, 0.5)
    assert m2 == pytest.approx(25.0)
    assert area_hectares(m2) == pytest.approx(0.0025)
    # 40,000 m2 is exactly 4 hectares.
    assert area_hectares(40_000.0) == pytest.approx(4.0)


def test_mask_iou_half_overlap_is_one_third() -> None:
    """Two equal masks sharing half their area have IoU = 1/3.

    Mask A covers columns 0..1, mask B covers columns 1..2 of a 1x3 grid; each
    has area 2, they overlap in column 1 (area 1), and their union is columns
    0..2 (area 3). IoU = 1 / 3.
    """
    a = np.array([[1, 1, 0]])
    b = np.array([[0, 1, 1]])
    assert mask_iou(a, b) == pytest.approx(1.0 / 3.0)
    # Identical masks score 1.0; two empty masks agree perfectly -> 1.0.
    assert mask_iou(a, a) == pytest.approx(1.0)
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == pytest.approx(1.0)


def test_invalid_inputs_raise() -> None:
    """Bad shapes, empties, and bad parameters raise ValueError."""
    with pytest.raises(ValueError):
        label_components(np.array([1, 1, 1]))  # 1-D
    with pytest.raises(ValueError):
        label_components(np.zeros((0, 0)))  # empty
    with pytest.raises(ValueError):
        label_components(np.ones((2, 2)), connectivity=6)  # bad connectivity
    with pytest.raises(ValueError):
        pixels_to_area(10, 0.0)  # non-positive pixel size
    with pytest.raises(ValueError):
        mask_iou(np.ones((2, 2)), np.ones((3, 3)))  # shape mismatch
    with pytest.raises(ValueError):
        filter_by_area([], min_px=10, max_px=5)  # min > max
