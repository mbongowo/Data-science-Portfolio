"""Pure-numpy raster mask analytics: the tested core of this project.

This module takes a binary or integer-labelled raster mask (e.g. the output of
Segment Anything applied to satellite imagery) and turns it into *counted,
measured* features: connected-component labelling, per-object properties (area,
bounding box, centroid), area filtering to drop noise and oversized blobs, and
conversions from pixels to square metres and hectares. A small
intersection-over-union helper supports validation against a reference mask.

Everything here depends on numpy and the standard library only. There is no
rasterio, shapely, geopandas, samgeo, or torch in this module, so it imports
and runs anywhere numpy is installed and is fully unit-tested. The geospatial
vectorisation and the SAM segmentation live in separate, lazily-imported
wrapper modules.

Connectivity convention
------------------------
* ``connectivity=4`` connects a pixel to its up/down/left/right neighbours.
* ``connectivity=8`` additionally connects the four diagonal neighbours.

Mask convention: any non-zero pixel is foreground; zero is background.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Neighbour offsets (row, col). The first four are the 4-connectivity set; all
# eight are the 8-connectivity set.
_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFFSETS_8 = _OFFSETS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _as_2d_binary(mask: np.ndarray) -> np.ndarray:
    """Validate and coerce ``mask`` into a 2-D boolean array.

    Raises ``ValueError`` if the input is not 2-D or has no elements.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("mask is empty")
    return arr != 0


def label_components(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Label connected foreground components with a BFS flood fill.

    Parameters
    ----------
    mask:
        2-D array. Any non-zero pixel is foreground.
    connectivity:
        ``4`` (orthogonal neighbours) or ``8`` (orthogonal + diagonal).

    Returns
    -------
    numpy.ndarray
        Integer array the same shape as ``mask``. Background is ``0``;
        components are numbered ``1, 2, ...`` in raster (row-major) scan order
        of their first-encountered pixel.

    Raises
    ------
    ValueError
        If ``mask`` is not 2-D, is empty, or ``connectivity`` is not 4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    fg = _as_2d_binary(mask)
    offsets = _OFFSETS_4 if connectivity == 4 else _OFFSETS_8
    rows, cols = fg.shape
    labels = np.zeros((rows, cols), dtype=np.int64)
    next_label = 0

    for r0 in range(rows):
        for c0 in range(cols):
            if not fg[r0, c0] or labels[r0, c0] != 0:
                continue
            next_label += 1
            # Iterative BFS so deep/large blobs never blow the recursion stack.
            stack = [(r0, c0)]
            labels[r0, c0] = next_label
            while stack:
                r, c = stack.pop()
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and fg[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
    return labels


def region_props(labeled: np.ndarray) -> list[dict[str, Any]]:
    """Compute per-object properties from a labelled array.

    Parameters
    ----------
    labeled:
        2-D integer array as returned by :func:`label_components` (``0`` is
        background).

    Returns
    -------
    list of dict
        One dict per object, ordered by ascending label, with keys:

        * ``label`` — the integer label.
        * ``area_px`` — pixel count.
        * ``bbox`` — ``(rmin, cmin, rmax, cmax)``, inclusive pixel indices.
        * ``centroid`` — ``(row, col)`` mean pixel coordinates (floats).

    Raises
    ------
    ValueError
        If ``labeled`` is not 2-D or is empty.
    """
    arr = np.asarray(labeled)
    if arr.ndim != 2:
        raise ValueError(f"labeled must be 2-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("labeled is empty")

    props: list[dict[str, Any]] = []
    labels = np.unique(arr)
    for lab in labels:
        if lab == 0:
            continue
        rr, cc = np.nonzero(arr == lab)
        props.append(
            {
                "label": int(lab),
                "area_px": int(rr.size),
                "bbox": (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())),
                "centroid": (float(rr.mean()), float(cc.mean())),
            }
        )
    return props


def count_objects(labeled: np.ndarray) -> int:
    """Return the number of distinct foreground objects in ``labeled``.

    Counts unique non-zero labels. Raises ``ValueError`` on a non-2-D or empty
    input.
    """
    arr = np.asarray(labeled)
    if arr.ndim != 2:
        raise ValueError(f"labeled must be 2-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("labeled is empty")
    return int(np.count_nonzero(np.unique(arr) != 0))


def filter_by_area(
    props: list[dict[str, Any]],
    min_px: int | None = None,
    max_px: int | None = None,
) -> list[dict[str, Any]]:
    """Keep only objects whose ``area_px`` falls in ``[min_px, max_px]``.

    Either bound may be ``None`` (open-ended). Bounds are inclusive. This is how
    we drop single-pixel speckle (``min_px``) and oversized blobs such as a
    merged field that swallowed several buildings (``max_px``).

    Parameters
    ----------
    props:
        A list of property dicts as returned by :func:`region_props`.
    min_px, max_px:
        Inclusive lower / upper area bounds in pixels, or ``None``.

    Returns
    -------
    list of dict
        The subset of ``props`` satisfying the bounds, original order kept.

    Raises
    ------
    ValueError
        If both bounds are given and ``min_px > max_px``.
    """
    if min_px is not None and max_px is not None and min_px > max_px:
        raise ValueError(f"min_px ({min_px}) must not exceed max_px ({max_px})")
    out = []
    for p in props:
        a = p["area_px"]
        if min_px is not None and a < min_px:
            continue
        if max_px is not None and a > max_px:
            continue
        out.append(p)
    return out


def pixels_to_area(area_px: float, pixel_size_m: float) -> float:
    """Convert a pixel count to square metres.

    Each pixel is assumed square with side ``pixel_size_m`` metres, so the area
    is ``area_px * pixel_size_m**2``.

    Raises ``ValueError`` if ``pixel_size_m`` is not positive.
    """
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be > 0, got {pixel_size_m}")
    return float(area_px) * float(pixel_size_m) ** 2


def area_hectares(area_m2: float) -> float:
    """Convert square metres to hectares (1 ha = 10,000 m**2)."""
    return float(area_m2) / 10_000.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two binary masks.

    Both inputs are treated as binary (non-zero is foreground) and must share a
    shape. IoU is ``|A and B| / |A or B|``. By convention two empty masks score
    ``1.0`` (they agree perfectly that nothing is foreground).

    Raises
    ------
    ValueError
        If the inputs are not 2-D, are empty, or differ in shape.
    """
    fa = _as_2d_binary(a)
    fb = _as_2d_binary(b)
    if fa.shape != fb.shape:
        raise ValueError(f"shape mismatch: {fa.shape} vs {fb.shape}")
    inter = int(np.count_nonzero(fa & fb))
    union = int(np.count_nonzero(fa | fb))
    if union == 0:
        return 1.0
    return inter / union
