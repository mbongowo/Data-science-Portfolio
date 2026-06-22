"""Pure-numpy before/after flood-extent change and hectares.

Given two boolean water masks — a *pre*-flood mask and a *post*-flood mask, each
from Otsu-thresholding the corresponding SAR scene — this module splits the scene
into the three meaningful flood classes and converts the pixel counts to
hectares:

* **flooded**         — water *after* but not *before* (post & ~pre): the new
  flood extent, the number that matters for a situation report.
* **permanent_water** — water in *both* (pre & post): rivers, lakes, the channel
  that is always wet and must not be counted as flood.
* **receded**         — water *before* but not *after* (pre & ~post): drainage /
  recession between the two dates.

Separating permanent water from new flood is the whole point of the before/after
design: a single post-flood water mask cannot tell a flooded field from the river
next to it. Everything here depends on numpy and the standard library only.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _check_bool_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equal-shaped masks; return them as boolean arrays."""
    ba = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if ba.shape != bb.shape:
        raise ValueError(f"shape mismatch: {ba.shape} vs {bb.shape}")
    return ba, bb


def flood_extent(
    pre_water: np.ndarray, post_water: np.ndarray
) -> dict[str, np.ndarray]:
    """Split two water masks into flooded / permanent / receded boolean masks.

    Parameters
    ----------
    pre_water:
        Boolean water mask before the flood.
    post_water:
        Boolean water mask after the flood. Same shape as ``pre_water``.

    Returns
    -------
    dict
        Boolean masks under keys:

        * ``flooded``         — ``post & ~pre`` (newly water),
        * ``permanent_water`` — ``pre & post`` (water at both dates),
        * ``receded``         — ``pre & ~post`` (water before, dry after).

    Raises
    ------
    ValueError
        If the two masks differ in shape.
    """
    pre, post = _check_bool_pair(pre_water, post_water)
    return {
        "flooded": post & ~pre,
        "permanent_water": pre & post,
        "receded": pre & ~post,
    }


def flood_stats(masks: dict[str, np.ndarray], pixel_size_m: float) -> dict[str, Any]:
    """Convert the flood-extent masks to pixel counts, hectares, and a fraction.

    Parameters
    ----------
    masks:
        The dict returned by :func:`flood_extent` (keys ``flooded``,
        ``permanent_water``, ``receded``).
    pixel_size_m:
        Pixel side length in metres (10 for Sentinel-1 GRD). One pixel is
        ``pixel_size_m ** 2`` square metres; 10 000 m2 = 1 hectare.

    Returns
    -------
    dict
        Keys: ``flooded_pixels``, ``flooded_hectares``,
        ``permanent_water_hectares``, ``receded_hectares``,
        ``flooded_fraction_of_scene`` (flooded pixels / total pixels).

    Raises
    ------
    ValueError
        If ``pixel_size_m`` is not positive or a required mask is missing.
    """
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be > 0, got {pixel_size_m}")
    for key in ("flooded", "permanent_water", "receded"):
        if key not in masks:
            raise ValueError(f"masks is missing required key {key!r}")

    flooded = np.asarray(masks["flooded"], dtype=bool)
    permanent = np.asarray(masks["permanent_water"], dtype=bool)
    receded = np.asarray(masks["receded"], dtype=bool)

    px_area_ha = float(pixel_size_m) ** 2 / 10_000.0
    flooded_pixels = int(np.count_nonzero(flooded))
    total = int(flooded.size)
    return {
        "flooded_pixels": flooded_pixels,
        "flooded_hectares": flooded_pixels * px_area_ha,
        "permanent_water_hectares": int(np.count_nonzero(permanent)) * px_area_ha,
        "receded_hectares": int(np.count_nonzero(receded)) * px_area_ha,
        "flooded_fraction_of_scene": flooded_pixels / total if total else 0.0,
    }
