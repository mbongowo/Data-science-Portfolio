"""Rasterize SpaceNet building footprints into segmentation masks.

Pairs each image GeoTIFF with its building-footprint GeoJSON, burns the polygons
onto the image's pixel grid, and writes the layout GeoSegDataModule expects:

    <out>/images/<stem>.tif
    <out>/masks/<stem>.tif      # single-band uint8, 1 = building, 0 = background

The image and its mask share a filename so the data module pairs them, and the
mask uses the image's CRS, transform, width, and height so the two line up pixel
for pixel. Run::

    python scripts/rasterize_spacenet.py \
        --images data/raw/RGB-PanSharpen \
        --labels data/raw/geojson/buildings \
        --out data/spacenet

Then train against it with ``data.data_dir=data/spacenet``.

rasterio and geopandas are imported lazily inside the functions that need them,
so the pure file-pairing helpers (and their test) run on a machine without the
geospatial stack.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

__all__ = ["extract_key", "pair_files", "rasterize_pair", "main"]

#: Default key regex. SpaceNet names files like ``RGB-PanSharpen_..._img1.tif``
#: and ``buildings_..._img1.geojson``; the shared ``img<N>`` token pairs them.
DEFAULT_KEY_REGEX = r"(img\d+)$"
IMAGE_EXTS = (".tif", ".tiff")
LABEL_EXTS = (".geojson", ".json")


def extract_key(stem: str, key_regex: str = DEFAULT_KEY_REGEX) -> str:
    """Return the pairing key for a filename stem.

    If ``key_regex`` matches, the last captured group (or the whole match if the
    pattern has no group) is the key. Otherwise the whole stem is the key. Two
    files that yield the same key are treated as an image/label pair.

    Parameters
    ----------
    stem : str
        Filename without directory or extension.
    key_regex : str, optional
        Regex applied to the stem. Defaults to the SpaceNet ``img<N>`` token.

    Returns
    -------
    str
        The pairing key.
    """
    match = re.search(key_regex, stem)
    if match:
        return match.group(match.lastindex) if match.lastindex else match.group(0)
    return stem


def pair_files(image_paths, label_paths, key_regex: str = DEFAULT_KEY_REGEX):
    """Pair image and label paths that share an extracted key.

    Parameters
    ----------
    image_paths, label_paths : iterable of str or Path
        Candidate image and label files.
    key_regex : str, optional
        Passed to :func:`extract_key`.

    Returns
    -------
    list of tuple
        ``(image_path, label_path, stem)`` for each matched image, ordered by
        stem. Images with no matching label are dropped. If two labels share a
        key the first one wins.
    """
    labels_by_key: dict[str, Path] = {}
    for path in label_paths:
        labels_by_key.setdefault(extract_key(Path(path).stem, key_regex), Path(path))

    pairs = []
    for path in image_paths:
        image = Path(path)
        label = labels_by_key.get(extract_key(image.stem, key_regex))
        if label is not None:
            pairs.append((image, label, image.stem))
    pairs.sort(key=lambda item: item[2])
    return pairs


def rasterize_pair(
    image_path,
    label_path,
    out_images,
    out_masks,
    *,
    all_touched: bool = False,
    link: bool = False,
):
    """Rasterize one image/label pair into the ``images/`` and ``masks/`` layout.

    The mask is burned onto the image's grid (same shape, transform, and CRS),
    with polygons reprojected to the image CRS first if they differ. An empty or
    building-free label produces an all-zero mask.

    Parameters
    ----------
    image_path, label_path : str or Path
        The image GeoTIFF and its building GeoJSON.
    out_images, out_masks : str or Path
        Output directories for the copied image and the written mask.
    all_touched : bool, optional
        Burn every pixel a polygon touches, not only those whose centre falls
        inside it. Defaults to False.
    link : bool, optional
        Hardlink the image instead of copying it, to save disk. Falls back to a
        copy if the link cannot be made (for example across filesystems).

    Returns
    -------
    tuple of Path
        ``(image_out_path, mask_out_path)``.
    """
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    stem = Path(image_path).stem
    out_images = Path(out_images)
    out_masks = Path(out_masks)
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    gdf = gpd.read_file(label_path)
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]

    if geoms:
        mask = rasterize(
            ((geom, 1) for geom in geoms),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        )
    else:
        mask = np.zeros((height, width), dtype="uint8")

    mask_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    mask_out = out_masks / f"{stem}.tif"
    with rasterio.open(mask_out, "w", **mask_profile) as dst:
        dst.write(mask, 1)

    image_out = out_images / f"{stem}.tif"
    if link:
        import os

        try:
            if image_out.exists():
                image_out.unlink()
            os.link(image_path, image_out)
        except OSError:
            shutil.copy2(image_path, image_out)
    else:
        shutil.copy2(image_path, image_out)

    return image_out, mask_out


def _list_files(directory, exts) -> list[Path]:
    """Return sorted files in ``directory`` whose suffix is in ``exts``."""
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", required=True, help="Directory of image GeoTIFFs.")
    parser.add_argument("--labels", required=True, help="Directory of building GeoJSON files.")
    parser.add_argument("--out", default="data/spacenet", help="Output dataset directory.")
    parser.add_argument(
        "--key-regex",
        default=DEFAULT_KEY_REGEX,
        help="Regex whose capture group pairs an image with a label "
        "(default matches the SpaceNet 'img<N>' token).",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Burn every pixel a polygon touches, not just pixel centres.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Hardlink images instead of copying them (saves disk).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N pairs (0 = all). Useful for a quick trial.",
    )
    args = parser.parse_args()

    images = _list_files(args.images, IMAGE_EXTS)
    labels = _list_files(args.labels, LABEL_EXTS)
    if not images:
        raise SystemExit(f"No image GeoTIFFs (.tif/.tiff) found in {args.images}")
    if not labels:
        raise SystemExit(f"No GeoJSON labels (.geojson/.json) found in {args.labels}")

    pairs = pair_files(images, labels, args.key_regex)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(
            "No image/label pairs matched. Check the file names and --key-regex."
        )

    out = Path(args.out)
    skipped = len(images) - len(pairs)
    print(
        f"Rasterizing {len(pairs)} image/label pairs into {out} "
        f"({skipped} images had no matching label)."
    )
    for i, (image, label, _stem) in enumerate(pairs, start=1):
        rasterize_pair(
            image,
            label,
            out / "images",
            out / "masks",
            all_touched=args.all_touched,
            link=args.link,
        )
        if i % 50 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} done")

    print(
        f"Done. Wrote {out}/images and {out}/masks. "
        f"Train with: python -m geoseg.train data.data_dir={out}"
    )


if __name__ == "__main__":
    main()
