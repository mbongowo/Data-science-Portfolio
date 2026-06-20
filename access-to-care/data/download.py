"""Reproducibly fetch all pipeline inputs into ``data/raw`` (git-ignored).

Sources and URLs are read from ``config/sources.yaml`` so they are configuration,
not hard-coded values. Run from the repository root::

    python data/download.py
    python data/download.py --only osm facilities

Datasets:
  * Geofabrik OSM extract for Cameroon (roads + amenities)
  * Healthsites.io cleaned health-facility points
  * WorldPop 100m gridded population
  * GADM admin-2 boundaries

Large files (OSM ~hundreds of MB, WorldPop raster) require a network connection;
existing files are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "sources.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"

CHUNK = 1 << 20  # 1 MiB


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML sources configuration."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _stream_to_file(url: str, dest: Path, params: dict[str, Any] | None = None) -> None:
    """Stream a URL to ``dest`` with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, params=params, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)


def download_simple(spec: dict[str, Any], force: bool) -> Path:
    """Download a single-file source described by ``spec``."""
    dest = RAW_DIR / spec["filename"]
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already present")
        return dest
    print(f"  [get ] {spec['name']} -> {dest.name}")
    _stream_to_file(spec["url"], dest)
    return dest


def download_facilities(spec: dict[str, Any], force: bool) -> Path:
    """Download Healthsites.io facilities, following pagination into one GeoJSON."""
    dest = RAW_DIR / spec["filename"]
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already present")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(spec.get("api_key_env", ""), "")
    base_url = spec["url"]
    features: list[dict[str, Any]] = []

    print(f"  [get ] {spec['name']} -> {dest.name}")
    page = 1
    while True:
        params = {"api-key": api_key} if api_key else None
        url = base_url.replace("page=1", f"page={page}")
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        payload = resp.json()
        page_features = payload.get("features", []) if isinstance(payload, dict) else payload
        if not page_features:
            break
        features.extend(page_features)
        print(f"    page {page}: +{len(page_features)} (total {len(features)})")
        if not spec.get("paginate", False):
            break
        page += 1

    collection = {"type": "FeatureCollection", "features": features}
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(collection, fh)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download access-to-care inputs.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["osm", "facilities", "population", "admin"],
        help="Subset of sources to fetch (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    sources = cfg["sources"]
    wanted = args.only or list(sources.keys())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {RAW_DIR}")

    for key in wanted:
        spec = sources[key]
        try:
            if key == "facilities":
                download_facilities(spec, args.force)
            else:
                download_simple(spec, args.force)
        except requests.RequestException as exc:  # pragma: no cover - network path
            print(f"  [fail] {key}: {exc}", file=sys.stderr)
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
