"""LightningDataModule for tiled EO semantic segmentation.

Design notes
------------
The tiling and deterministic-split logic is deliberately implemented as
*pure-python* helpers (:func:`compute_tile_grid`, :func:`deterministic_split`)
so they can be unit-tested without torch installed. The Lightning/torch pieces
(``Dataset``, ``LightningDataModule``) import their heavy deps lazily inside the
class bodies, so ``import geoseg.datamodule`` succeeds on a bare machine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "TileSpec",
    "compute_tile_grid",
    "deterministic_split",
    "normalize_per_band",
    "GeoSegDataset",
    "GeoSegDataModule",
]


@dataclass(frozen=True)
class TileSpec:
    """A single tile window into a larger raster (row/col in pixels)."""

    row: int
    col: int
    height: int
    width: int

    @property
    def key(self) -> str:
        return f"{self.row}_{self.col}_{self.height}_{self.width}"


def compute_tile_grid(
    raster_height: int,
    raster_width: int,
    tile_size: int,
    stride: int | None = None,
    drop_partial: bool = True,
) -> list[TileSpec]:
    """Compute a deterministic grid of tile windows over a raster.

    Parameters
    ----------
    raster_height, raster_width:
        Size of the source raster in pixels.
    tile_size:
        Square tile edge length in pixels.
    stride:
        Step between tile origins. Defaults to ``tile_size`` (no overlap).
    drop_partial:
        If True, tiles that would run off the raster edge are dropped. If False,
        the final row/column is snapped back so the tile stays in-bounds (which
        can produce overlapping edge tiles).

    Returns
    -------
    list[TileSpec]
        Ordered, reproducible list of tile windows.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if stride is None:
        stride = tile_size
    if stride <= 0:
        raise ValueError("stride must be positive")

    tiles: list[TileSpec] = []
    for row in range(0, max(raster_height, 1), stride):
        for col in range(0, max(raster_width, 1), stride):
            r, c = row, col
            if r + tile_size > raster_height:
                if drop_partial:
                    continue
                r = max(raster_height - tile_size, 0)
            if c + tile_size > raster_width:
                if drop_partial:
                    continue
                c = max(raster_width - tile_size, 0)
            tiles.append(TileSpec(r, c, tile_size, tile_size))
    # Deduplicate (snapping can create repeats) while preserving order.
    seen: set[str] = set()
    unique: list[TileSpec] = []
    for t in tiles:
        if t.key not in seen:
            seen.add(t.key)
            unique.append(t)
    return unique


def _stable_hash(value: str, seed: int) -> float:
    """Deterministic float in [0, 1) from a string + seed (hash-based split)."""
    h = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    # Use the first 8 hex chars -> 32-bit int -> [0, 1).
    return int(h[:8], 16) / 0xFFFFFFFF


def deterministic_split(
    keys: list[str],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministically assign keys to train/val/test by stable hashing.

    The assignment depends only on the key string and the seed, so it is stable
    across machines and runs and does not require holding all data in memory.

    Returns a dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    if len(fractions) != 3:
        raise ValueError("fractions must be a 3-tuple (train, val, test)")
    total = sum(fractions)
    if total <= 0:
        raise ValueError("fractions must sum to a positive value")
    f_train, f_val, _ = (f / total for f in fractions)
    cut_train = f_train
    cut_val = f_train + f_val

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in keys:
        r = _stable_hash(key, seed)
        if r < cut_train:
            split["train"].append(key)
        elif r < cut_val:
            split["val"].append(key)
        else:
            split["test"].append(key)
    return split


def normalize_per_band(
    image: np.ndarray,
    means: np.ndarray | list[float],
    stds: np.ndarray | list[float],
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-band standardisation. ``image`` is (C, H, W); means/stds length C."""
    img = np.asarray(image, dtype=np.float32)
    means = np.asarray(means, dtype=np.float32).reshape(-1, 1, 1)
    stds = np.asarray(stds, dtype=np.float32).reshape(-1, 1, 1)
    if img.shape[0] != means.shape[0]:
        raise ValueError(
            f"band count {img.shape[0]} != stats length {means.shape[0]}"
        )
    return (img - means) / (stds + eps)


def _make_dataset_base():
    """Return a torch Dataset subclass (imported lazily)."""
    from torch.utils.data import Dataset  # noqa: PLC0415

    return Dataset


class GeoSegDataset:
    """Tile-backed segmentation dataset (image, mask) over rasters.

    Heavy deps (torch, rasterio, albumentations) are imported lazily. Each item
    yields a dict with float32 ``image`` (C,H,W) and float32 ``mask`` (1,H,W)
    tensors.
    """

    def __init__(
        self,
        samples: list[dict],
        tiles: list[TileSpec],
        band_means: list[float],
        band_stds: list[float],
        transform=None,
    ) -> None:
        self.samples = samples
        self.tiles = tiles
        self.band_means = band_means
        self.band_stds = band_stds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _read_window(self, path: str, tile: TileSpec) -> np.ndarray:
        import rasterio  # noqa: PLC0415
        from rasterio.windows import Window  # noqa: PLC0415

        with rasterio.open(path) as src:
            window = Window(tile.col, tile.row, tile.width, tile.height)
            return src.read(window=window)

    def __getitem__(self, idx: int):  # pragma: no cover - needs torch+rasterio
        import torch  # noqa: PLC0415

        sample = self.samples[idx]
        tile = self.tiles[idx % len(self.tiles)] if self.tiles else None

        image = self._read_window(sample["image"], tile) if tile else None
        mask = self._read_window(sample["mask"], tile) if tile else None
        image = normalize_per_band(image, self.band_means, self.band_stds)
        mask = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        if self.transform is not None:
            # albumentations expects HWC image and HW mask.
            augmented = self.transform(
                image=np.transpose(image, (1, 2, 0)),
                mask=mask[0],
            )
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = augmented["mask"][None, ...]

        return {
            "image": torch.as_tensor(image, dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.float32),
        }


class GeoSegDataModule:
    """LightningDataModule wrapping :class:`GeoSegDataset`.

    Inherits from ``pytorch_lightning.LightningDataModule`` at runtime (resolved
    lazily) so that importing this module does not require lightning.
    """

    def __new__(cls, *args, **kwargs):
        # Dynamically rebase onto LightningDataModule the first time we are
        # instantiated, keeping import side-effect free.
        try:
            import pytorch_lightning as pl  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs lightning
            raise ImportError(
                "GeoSegDataModule requires pytorch_lightning. Install the full "
                "environment (pixi install) to use the training pipeline."
            ) from exc
        if pl.LightningDataModule not in cls.__bases__:
            cls.__bases__ = (pl.LightningDataModule, *cls.__bases__)
        return super().__new__(cls)

    def __init__(
        self,
        data_dir: str,
        tile_size: int = 256,
        stride: int | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        band_means: list[float] | None = None,
        band_stds: list[float] | None = None,
        split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.band_means = band_means or [0.0, 0.0, 0.0]
        self.band_stds = band_stds or [1.0, 1.0, 1.0]
        self.split_fractions = split_fractions
        self.seed = seed
        self._splits: dict[str, list[str]] = {}
        self._datasets: dict[str, GeoSegDataset] = {}

    # --- augmentation ---------------------------------------------------
    def _train_transform(self):  # pragma: no cover - needs albumentations
        import albumentations as A  # noqa: PLC0415

        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )

    # --- lightning hooks ------------------------------------------------
    def setup(self, stage: str | None = None) -> None:  # pragma: no cover
        from pathlib import Path  # noqa: PLC0415

        image_dir = Path(self.data_dir) / "images"
        mask_dir = Path(self.data_dir) / "masks"
        pairs = []
        for img in sorted(image_dir.glob("*.tif")):
            mask = mask_dir / img.name
            if mask.exists():
                pairs.append({"image": str(img), "mask": str(mask), "key": img.stem})

        self._splits = deterministic_split(
            [p["key"] for p in pairs],
            fractions=self.split_fractions,
            seed=self.seed,
        )
        by_key = {p["key"]: p for p in pairs}
        # A representative tile grid (single full-size tile per image here).
        tiles = compute_tile_grid(self.tile_size, self.tile_size, self.tile_size)
        for name, keys in self._splits.items():
            samples = [by_key[k] for k in keys]
            transform = self._train_transform() if name == "train" else None
            self._datasets[name] = GeoSegDataset(
                samples=samples,
                tiles=tiles,
                band_means=self.band_means,
                band_stds=self.band_stds,
                transform=transform,
            )

    def _loader(self, name: str, shuffle: bool):  # pragma: no cover - needs torch
        from torch.utils.data import DataLoader  # noqa: PLC0415

        return DataLoader(
            self._datasets[name],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=shuffle,
        )

    def train_dataloader(self):  # pragma: no cover - needs torch
        return self._loader("train", shuffle=True)

    def val_dataloader(self):  # pragma: no cover - needs torch
        return self._loader("val", shuffle=False)

    def test_dataloader(self):  # pragma: no cover - needs torch
        return self._loader("test", shuffle=False)
