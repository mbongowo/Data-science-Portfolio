"""Pure-numpy sliding-window tiling and stitching.

Large rasters do not fit through a fixed-size network in one pass, so they are
cut into overlapping square windows, predicted tile-by-tile, and reassembled.
These helpers cover the geometry of that process and depend only on
:mod:`numpy`, so they import and test without torch/rasterio.

:func:`tile_indices` enumerates the windows; :func:`stitch` reassembles per-tile
arrays back into a full image, averaging any overlap so seams blend smoothly.
The complementary grid generator in :mod:`geoseg.datamodule`
(:func:`~geoseg.datamodule.compute_tile_grid`) returns ``TileSpec`` dataclasses
for dataset construction; this module instead speaks plain
``(row0, row1, col0, col1)`` windows that slice directly with ``arr[r0:r1, c0:c1]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tile_indices", "stitch"]


def tile_indices(
    height: int,
    width: int,
    tile: int,
    overlap: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Enumerate windows that cover an ``(height, width)`` image.

    Windows are square ``tile x tile`` and step by ``tile - overlap`` pixels.
    Every pixel is covered: when the stride does not divide the image evenly the
    final row/column of windows is snapped back to end exactly on the image edge
    (``row1 == height`` / ``col1 == width``), which makes the last windows
    overlap their neighbours rather than running out of bounds.

    Parameters
    ----------
    height, width : int
        Image size in pixels. Must be positive.
    tile : int
        Square window edge length. A ``tile`` larger than a dimension is clamped
        to that dimension, giving a single full-extent window along it.
    overlap : int, optional
        Pixels of overlap between adjacent windows; must be in
        ``[0, tile)``. ``0`` (default) gives a non-overlapping grid.

    Returns
    -------
    list of tuple
        Ordered, de-duplicated ``(row0, row1, col0, col1)`` windows. Slicing
        ``arr[row0:row1, col0:col1]`` yields the tile.

    Raises
    ------
    ValueError
        If ``height``/``width``/``tile`` are not positive or ``overlap`` is out
        of range.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile <= 0:
        raise ValueError("tile must be positive")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")

    th = min(tile, height)
    tw = min(tile, width)
    stride_r = th - overlap if th > overlap else th
    stride_c = tw - overlap if tw > overlap else tw

    def _starts(length: int, win: int, stride: int) -> list[int]:
        if win >= length:
            return [0]
        starts = list(range(0, length - win + 1, stride))
        last = length - win
        if starts[-1] != last:
            starts.append(last)
        return starts

    row_starts = _starts(height, th, stride_r)
    col_starts = _starts(width, tw, stride_c)

    windows: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0 in row_starts:
        for c0 in col_starts:
            win = (r0, r0 + th, c0, c0 + tw)
            if win not in seen:
                seen.add(win)
                windows.append(win)
    return windows


def stitch(
    tiles: list[np.ndarray],
    positions: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    """Reassemble per-tile arrays into a full image, averaging overlaps.

    The inverse of :func:`tile_indices`: each tile is added into the output at
    its window and a parallel count tracks how many tiles touched every pixel,
    so overlapping regions are divided by their coverage and become the mean of
    the contributing tiles. With ``overlap=0`` this is an exact round-trip.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Tile arrays. Each is either 2-D ``(h, w)`` or has trailing channels
        ``(h, w, c)``; all tiles must agree on channel layout.
    positions : list of tuple
        ``(row0, row1, col0, col1)`` window for each tile, as returned by
        :func:`tile_indices`. Must be the same length as ``tiles``.
    height, width : int
        Size of the reconstructed image.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(height, width)`` or ``(height, width, c)``.

    Raises
    ------
    ValueError
        If ``tiles`` and ``positions`` differ in length, ``tiles`` is empty, or
        a tile's shape does not match its window.
    """
    if len(tiles) != len(positions):
        raise ValueError("tiles and positions must have the same length")
    if not tiles:
        raise ValueError("need at least one tile to stitch")

    first = np.asarray(tiles[0])
    channels = first.shape[2:] if first.ndim > 2 else ()
    out_shape = (height, width, *channels)
    acc = np.zeros(out_shape, dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)

    for tile, (r0, r1, c0, c1) in zip(tiles, positions, strict=True):
        arr = np.asarray(tile, dtype=np.float64)
        if arr.shape[:2] != (r1 - r0, c1 - c0):
            raise ValueError(
                f"tile shape {arr.shape[:2]} does not match window "
                f"{(r1 - r0, c1 - c0)}"
            )
        acc[r0:r1, c0:c1] += arr
        count[r0:r1, c0:c1] += 1.0

    if (count == 0).any():
        raise ValueError("positions do not cover every pixel")
    if channels:
        count = count.reshape(height, width, *([1] * len(channels)))
    return acc / count
