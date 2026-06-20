"""Config loading & validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eo_monitor.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CORN_BELT = REPO_ROOT / "config" / "corn_belt.yaml"


def test_load_corn_belt_config() -> None:
    cfg = load_config(CORN_BELT)
    assert isinstance(cfg, Config)
    assert cfg.indices == ["NDVI", "NDWI", "NDMI"]
    assert cfg.aoi.bbox is not None and len(cfg.aoi.bbox) == 4
    assert cfg.date_range.start.year == 2023
    assert cfg.baseline.months == [7, 8]
    assert cfg.cloud_cover_max == 20


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "config" / "does_not_exist.yaml")


def test_invalid_index_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "aoi": {"bbox": [-97.1, 41.05, -96.85, 41.25]},
                "date_range": {"start": "2023-07-01", "end": "2023-08-31"},
                "baseline": {"start": "2019-07-01", "end": "2022-08-31"},
                "indices": ["EVI"],
            }
        )


def test_bbox_and_vector_both_set_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "aoi": {"bbox": [-97.1, 41.05, -96.85, 41.25], "vector_path": "x.geojson"},
                "date_range": {"start": "2023-07-01", "end": "2023-08-31"},
                "baseline": {"start": "2019-07-01", "end": "2022-08-31"},
                "indices": ["NDVI"],
            }
        )


def test_reversed_date_range_raises() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "aoi": {"bbox": [-97.1, 41.05, -96.85, 41.25]},
                "date_range": {"start": "2023-09-01", "end": "2023-08-31"},
                "baseline": {"start": "2019-07-01", "end": "2022-08-31"},
                "indices": ["NDVI"],
            }
        )
