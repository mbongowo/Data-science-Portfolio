"""Validate detected disturbances against a known event.

Given a per-pixel map of detected breakpoint *dates* and *magnitudes* and a
documented event (a reference date plus a polygon/mask of the affected area),
we compute a simple spatial-agreement summary:

* **detection rate** - fraction of the event area where a disturbance was
  detected within a time window of the event date and with a magnitude beyond
  a (negative) threshold;
* **false-alarm rate** - fraction of pixels *outside* the event area that were
  nonetheless flagged in the same window.

The numpy core works on plain arrays so it can be tested and reasoned about
without GIS dependencies; helpers to rasterise a polygon are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ValidationResult", "spatial_agreement", "rasterize_event", "summary"]


@dataclass
class ValidationResult:
    """Spatial agreement between detections and a known event."""

    n_event_pixels: int
    n_detected_in_event: int
    detection_rate: float
    n_outside_pixels: int
    n_false_alarms: int
    false_alarm_rate: float
    window_days: int
    event_date: np.datetime64

    def __str__(self) -> str:
        return (
            f"Event {self.event_date} (+/-{self.window_days} d):\n"
            f"  detection rate : {self.detection_rate:6.1%} "
            f"({self.n_detected_in_event}/{self.n_event_pixels} event px)\n"
            f"  false-alarm    : {self.false_alarm_rate:6.1%} "
            f"({self.n_false_alarms}/{self.n_outside_pixels} background px)"
        )


def spatial_agreement(
    detected_dates: np.ndarray,
    detected_magnitude: np.ndarray,
    event_mask: np.ndarray,
    event_date: str | np.datetime64,
    window_days: int = 30,
    magnitude_threshold: float = -0.1,
) -> ValidationResult:
    """Compare detection maps to an event mask.

    Parameters
    ----------
    detected_dates:
        2-D array of detected breakpoint dates (``datetime64``); ``NaT`` where
        nothing was detected.
    detected_magnitude:
        2-D array of breakpoint magnitudes (same shape). Disturbances are
        *negative* (NDVI drops).
    event_mask:
        Boolean 2-D array, ``True`` inside the documented event footprint.
    event_date:
        Reference date of the documented event.
    window_days:
        A detection counts as agreeing if its date is within this many days of
        ``event_date``.
    magnitude_threshold:
        Maximum (most positive) magnitude that still counts as a real drop;
        detections must be ``<= magnitude_threshold``.
    """
    dates = np.asarray(detected_dates, dtype="datetime64[D]")
    mag = np.asarray(detected_magnitude, dtype=float)
    mask = np.asarray(event_mask, dtype=bool)
    ref = np.datetime64(event_date, "D")

    if not (dates.shape == mag.shape == mask.shape):
        raise ValueError("detected_dates, magnitude and mask must share shape")

    window = np.timedelta64(int(window_days), "D")
    delta = np.abs(dates - ref)

    flagged = (
        ~np.isnat(dates)
        & (delta <= window)
        & (mag <= magnitude_threshold)
    )

    n_event = int(mask.sum())
    n_det_in = int((flagged & mask).sum())
    n_out = int((~mask).sum())
    n_fa = int((flagged & ~mask).sum())

    return ValidationResult(
        n_event_pixels=n_event,
        n_detected_in_event=n_det_in,
        detection_rate=(n_det_in / n_event) if n_event else float("nan"),
        n_outside_pixels=n_out,
        n_false_alarms=n_fa,
        false_alarm_rate=(n_fa / n_out) if n_out else float("nan"),
        window_days=int(window_days),
        event_date=ref,
    )


def rasterize_event(geojson_path: str, like) -> np.ndarray:
    """Rasterise an event polygon onto the grid of a reference DataArray.

    Lazily imports ``rioxarray``/``rasterio``/``geopandas``. ``like`` is an
    ``xarray.DataArray`` (or anything with ``rio`` geobox) defining the target
    grid. Returns a boolean mask.
    """
    try:
        import geopandas as gpd
        from rasterio.features import geometry_mask
    except ImportError as exc:  # pragma: no cover - needs geo stack
        raise ImportError(
            "rasterize_event requires geopandas + rasterio (install via "
            "`pixi install`)."
        ) from exc

    gdf = gpd.read_file(geojson_path).to_crs(like.rio.crs)
    transform = like.rio.transform()
    out_shape = (like.rio.height, like.rio.width)
    mask = geometry_mask(
        gdf.geometry, out_shape=out_shape, transform=transform, invert=True
    )
    return mask


def summary(result: ValidationResult) -> str:
    """Print and return a human-readable summary of ``result``."""
    text = str(result)
    print(text)
    return text
