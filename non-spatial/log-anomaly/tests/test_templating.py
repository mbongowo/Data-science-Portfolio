"""Known-answer tests for Drain-lite templating.

These check the masking on a hand-worked example and the id-table stability,
with no third-party dependency, so they always execute.

Worked example:

    "Receiving block blk_123 src: /10.0.0.1:50010 size 4096"

    * blk_123          -> <*>   (block id)
    * /10.0.0.1:50010  -> <*>   (IPv4:port with leading slash)
    * 4096             -> <*>   (bare number)

    => "Receiving block <*> src: <*> size <*>"
"""

from __future__ import annotations

from loganomaly.templating import mask_line, template_id


def test_mask_line_worked_example() -> None:
    """The documented HDFS line masks to the expected template."""
    line = "Receiving block blk_123 src: /10.0.0.1:50010 size 4096"
    assert mask_line(line) == "Receiving block <*> src: <*> size <*>"


def test_mask_line_negative_block_id() -> None:
    """HDFS block ids carry an optional leading minus and still mask."""
    assert mask_line("PacketResponder blk_-9214 terminating") == (
        "PacketResponder <*> terminating"
    )


def test_mask_line_hex_id() -> None:
    """0x-prefixed and long bare hex runs mask to <*>."""
    assert mask_line("session 0xDEADBEEF opened") == "session <*> opened"


def test_mask_line_collapses_to_same_template() -> None:
    """Two lines that differ only in their variables share one template."""
    a = mask_line("Receiving block blk_1 src: /10.0.0.1:50010 size 4096")
    b = mask_line("Receiving block blk_2 src: /10.0.0.2:50011 size 8192")
    assert a == b


def test_template_id_is_stable_for_same_template() -> None:
    """The same template gets the same id; the table is reused across calls."""
    table: dict[str, int] = {}
    id_a = template_id("Receiving block blk_1 size 4096", table)
    id_b = template_id("Receiving block blk_2 size 8192", table)
    assert id_a == id_b == 0
    assert len(table) == 1


def test_template_id_assigns_new_id_for_new_template() -> None:
    """A genuinely different line gets the next id; ids are 0, 1, 2, ..."""
    table: dict[str, int] = {}
    first = template_id("Receiving block blk_1 size 4096", table)
    second = template_id("PacketResponder blk_1 terminating", table)
    third = template_id("Deleting block blk_9 file /tmp/x", table)
    assert (first, second, third) == (0, 1, 2)
    assert len(table) == 3
    # Re-seeing the first template returns its original id.
    assert template_id("Receiving block blk_5 size 16", table) == 0
