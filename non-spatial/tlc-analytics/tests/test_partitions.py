"""Known-answer + edge-case tests for the partition-path logic (pure stdlib)."""

from __future__ import annotations

import pytest

from tlc.partitions import iter_partitions, partition_relpath


def test_partition_relpath_pads_month() -> None:
    assert partition_relpath(2023, 1) == "year=2023/month=01"
    assert partition_relpath(2023, 12) == "year=2023/month=12"


def test_partition_relpath_rejects_bad_month() -> None:
    with pytest.raises(ValueError):
        partition_relpath(2023, 0)
    with pytest.raises(ValueError):
        partition_relpath(2023, 13)


def test_iter_partitions_grid_order() -> None:
    # 2 years x 2 months = 4 cells, years outer, months inner.
    got = list(iter_partitions("data/raw/yellow", [2022, 2023], [1, 2]))
    assert got == [
        "data/raw/yellow/year=2022/month=01",
        "data/raw/yellow/year=2022/month=02",
        "data/raw/yellow/year=2023/month=01",
        "data/raw/yellow/year=2023/month=02",
    ]


def test_iter_partitions_single_partition() -> None:
    got = list(iter_partitions("root", [2021], [7]))
    assert got == ["root/year=2021/month=07"]


def test_iter_partitions_count_is_product() -> None:
    got = list(iter_partitions("r", range(2019, 2024), range(1, 13)))
    assert len(got) == 5 * 12  # no dedup, exact product


def test_iter_partitions_strips_trailing_slash() -> None:
    got = list(iter_partitions("root/", [2023], [3]))
    assert got == ["root/year=2023/month=03"]


def test_iter_partitions_is_pure_no_io(tmp_path) -> None:
    # Paths are yielded for partitions that do not exist on disk: pure logic.
    missing_root = tmp_path / "does_not_exist"
    got = list(iter_partitions(str(missing_root).replace("\\", "/"), [2023], [1]))
    assert got[0].endswith("year=2023/month=01")
    assert not missing_root.exists()
