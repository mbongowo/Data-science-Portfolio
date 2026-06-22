"""Pydantic configuration models and YAML loading for eo-monitor.

The whole pipeline is driven by a single YAML file (see ``config/corn_belt.yaml``).
Malformed configuration raises a clear ``pydantic.ValidationError``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator

# Supported spectral indices. Keep in sync with eo_monitor.indices.
SUPPORTED_INDICES = ("NDVI", "NDWI", "NDMI")


class AOI(BaseModel):
    """Area of interest: either a bounding box OR a path to a vector file."""

    bbox: list[float] | None = Field(
        default=None,
        description="(lon_min, lat_min, lon_max, lat_max) in EPSG:4326.",
    )
    vector_path: Path | None = Field(
        default=None,
        description="Path to a vector file (GeoJSON/GPKG/SHP) defining the AOI.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> AOI:
        if (self.bbox is None) == (self.vector_path is None):
            raise ValueError("AOI must set exactly one of 'bbox' or 'vector_path'.")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must have 4 values: [lon_min, lat_min, lon_max, lat_max].")
            lon_min, lat_min, lon_max, lat_max = self.bbox
            if not (lon_min < lon_max and lat_min < lat_max):
                raise ValueError("bbox must satisfy lon_min<lon_max and lat_min<lat_max.")
            if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                raise ValueError("bbox longitudes must be in [-180, 180].")
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                raise ValueError("bbox latitudes must be in [-90, 90].")
        return self


class DateRange(BaseModel):
    """Inclusive observation window."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.start > self.end:
            raise ValueError(f"date_range.start ({self.start}) must be <= end ({self.end}).")
        return self

    def as_query(self) -> str:
        """Render an RFC 3339 datetime interval for STAC search."""
        return f"{self.start.isoformat()}/{self.end.isoformat()}"


class BaselineWindow(DateRange):
    """Climatological baseline window with an optional month filter."""

    months: list[Annotated[int, Field(ge=1, le=12)]] = Field(
        default_factory=lambda: list(range(1, 13))
    )


class StacConfig(BaseModel):
    """STAC endpoint and collection to search."""

    url: str = "https://earth-search.aws.element84.com/v1"
    collection: str = "sentinel-2-l2a"


class OutputConfig(BaseModel):
    """Where results are written and whether to also write PNG quicklooks."""

    dir: Path = Path("outputs")
    write_quicklook: bool = True


class Config(BaseModel):
    """Top-level pipeline configuration."""

    aoi: AOI
    date_range: DateRange
    baseline: BaselineWindow
    indices: list[str]
    cloud_cover_max: float = Field(default=20.0, ge=0, le=100)
    max_items: int = Field(default=60, gt=0)
    resolution: float = Field(default=20.0, gt=0)
    crs: str = "EPSG:32614"
    groupby: str = "solar_day"
    stac: StacConfig = Field(default_factory=StacConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _validate_indices(self) -> Config:
        if not self.indices:
            raise ValueError("At least one index must be requested.")
        unknown = [i for i in self.indices if i.upper() not in SUPPORTED_INDICES]
        if unknown:
            raise ValueError(
                f"Unsupported indices {unknown}. Supported: {list(SUPPORTED_INDICES)}."
            )
        # Normalise to canonical upper-case names.
        self.indices = [i.upper() for i in self.indices]
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pydantic.ValidationError
        If the YAML content does not satisfy the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return Config.model_validate(raw)
