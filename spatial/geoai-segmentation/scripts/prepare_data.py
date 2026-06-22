"""Download / prepare a building-footprint segmentation dataset.

Three modes:

1. ``--synthetic``  : generate a tiny synthetic GeoTIFF dataset so the pipeline
                      (and the smoke test) can run end-to-end with no downloads.
                      Requires rasterio + numpy.
2. ``--torchgeo``   : print instructions for using a torchgeo built-in dataset
                      (clean benchmark loaders) instead of raw SpaceNet.
3. (default)        : print SpaceNet access instructions (AWS Open Data).

SpaceNet access
---------------
SpaceNet building-footprint data lives in the AWS Open Data registry and is
free to access without credentials via the ``spacenet-dataset`` bucket. You need the AWS CLI:

    aws s3 ls s3://spacenet-dataset/ --no-sign-request
    aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/ ./data/raw/ \\
        --recursive --no-sign-request

Then convert the GeoJSON footprints to raster masks aligned to each image tile
(rasterio.features.rasterize) and place them under ``data/spacenet/images`` and
``data/spacenet/masks`` with matching filenames.

Google Open Buildings is an alternative vector source (CC-BY-4.0) covering the
Global South: https://sites.research.google/open-buildings/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def make_synthetic(out_dir: Path, n: int = 8, size: int = 256) -> None:
    """Generate ``n`` synthetic image/mask GeoTIFF pairs for smoke testing."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(n):
        img = (rng.random((3, size, size)) * 255).astype("uint8")
        mask = np.zeros((1, size, size), dtype="uint8")
        # a few random rectangular "buildings"
        for _ in range(rng.integers(3, 8)):
            r0, c0 = rng.integers(0, size - 40, size=2)
            h, w = rng.integers(15, 40, size=2)
            mask[0, r0 : r0 + h, c0 : c0 + w] = 1
            img[:, r0 : r0 + h, c0 : c0 + w] = rng.integers(120, 220)
        transform = from_origin(0, size, 1, 1)
        profile = dict(
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        )
        with rasterio.open(img_dir / f"tile_{i:03d}.tif", "w", **profile) as dst:
            dst.write(img)
        profile.update(count=1)
        with rasterio.open(mask_dir / f"tile_{i:03d}.tif", "w", **profile) as dst:
            dst.write(mask)
    print(f"Wrote {n} synthetic image/mask pairs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/spacenet", help="Output directory.")
    parser.add_argument("--synthetic", action="store_true", help="Make tiny data.")
    parser.add_argument("--n", type=int, default=8, help="Number of synthetic tiles.")
    parser.add_argument("--torchgeo", action="store_true", help="torchgeo hint.")
    args = parser.parse_args()

    if args.synthetic:
        make_synthetic(Path(args.out), n=args.n)
        return
    if args.torchgeo:
        print(
            "Use a torchgeo built-in dataset, e.g.:\n"
            "    from torchgeo.datasets import SpaceNet1\n"
            "    ds = SpaceNet1(root='data/raw', download=True)\n"
            "See https://torchgeo.readthedocs.io for available benchmark loaders."
        )
        return
    print(__doc__)


if __name__ == "__main__":
    main()
