"""Fetch validation-event polygons (known fires / deforestation events).

This is an *optional* helper: the analysis runs against live STAC imagery, but
to validate detections we need a documented event footprint. The default target
is an MTBS (Monitoring Trends in Burn Severity) fire perimeter, but any GeoJSON
URL works.

Usage
-----
    python data/download.py --url <geojson_url> --out data/raw/event.geojson

Or rely on the URL stored in ``config/aoi.yaml`` (validation.event_geojson is
the *destination*; pass the source URL explicitly).

No heavy dependencies: uses the standard library only.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Documented event portals. Replace or extend as needed. MTBS and GFW
# distribute perimeters; these placeholders point at their download pages.
KNOWN_EVENTS = {
    "creek-fire-2020": (
        "https://www.mtbs.gov/"  # MTBS direct-download portal (manual export)
    ),
}


def download(url: str, out_path: str | Path) -> Path:
    """Download ``url`` to ``out_path`` (creating parent dirs)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n        -> {out}")
    req = urllib.request.Request(url, headers={"User-Agent": "disturb/0.1"})
    with urllib.request.urlopen(req) as resp, open(out, "wb") as fh:
        fh.write(resp.read())
    print(f"Saved {out.stat().st_size} bytes")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        help="GeoJSON URL of the event perimeter to download.",
    )
    parser.add_argument(
        "--event",
        choices=sorted(KNOWN_EVENTS),
        help="Shortcut for a known event portal (see KNOWN_EVENTS).",
    )
    parser.add_argument(
        "--out",
        default="data/raw/event.geojson",
        help="Destination path (default: data/raw/event.geojson).",
    )
    args = parser.parse_args(argv)

    url = args.url
    if not url and args.event:
        url = KNOWN_EVENTS[args.event]
        print(
            f"Note: '{args.event}' resolves to a data portal ({url}); export "
            "the perimeter GeoJSON there, or pass a direct --url."
        )
        return 0
    if not url:
        parser.error("provide --url or --event")

    download(url, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
