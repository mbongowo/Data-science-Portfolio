"""Typer inference CLI.

Loads a trained checkpoint and predicts a binary segmentation mask for an
arbitrary input GeoTIFF tile, writing a *georeferenced* GeoTIFF whose spatial
profile (CRS, transform) is copied from the input.

Usage::

    python -m geoseg.infer run --checkpoint ckpt.ckpt --input tile.tif \\
        --output mask.tif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["app", "predict_mask", "main"]


def _lazy_typer():
    import typer  # noqa: PLC0415

    return typer


try:  # Build the app only if typer is present; keep import side-effect-free.
    import typer as _typer

    app = _typer.Typer(
        add_completion=False,
        help="geoseg inference: predict a segmentation mask on a new tile.",
    )
except ImportError:  # pragma: no cover - typer optional at import time
    app = None


def predict_mask(
    model,
    image: np.ndarray,
    band_means: list[float] | None = None,
    band_stds: list[float] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:  # pragma: no cover - needs torch
    """Predict a {0,1} uint8 mask (H, W) for a single (C, H, W) image array."""
    import torch  # noqa: PLC0415

    from geoseg.datamodule import normalize_per_band

    c = image.shape[0]
    means = band_means or [0.0] * c
    stds = band_stds or [1.0] * c
    norm = normalize_per_band(image, means, stds)
    tensor = torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))
    mask = (prob[0, 0].cpu().numpy() > threshold).astype(np.uint8)
    return mask


def _run_inference(
    checkpoint: str,
    input_path: str,
    output_path: str,
    threshold: float,
) -> str:  # pragma: no cover - needs torch + rasterio
    import rasterio  # noqa: PLC0415

    from geoseg.model import SegmentationModule

    model = SegmentationModule.load_from_checkpoint(checkpoint)

    with rasterio.open(input_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    mask = predict_mask(model, image, threshold=threshold)

    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask[None, ...])
    return output_path


if app is not None:

    @app.command()
    def run(  # pragma: no cover - CLI wiring
        checkpoint: str = _typer.Option(..., help="Path to a .ckpt file."),
        input: str = _typer.Option(..., help="Input GeoTIFF tile."),
        output: str = _typer.Option("prediction.tif", help="Output GeoTIFF."),
        threshold: float = _typer.Option(0.5, help="Probability threshold."),
    ) -> None:
        """Predict a georeferenced mask for INPUT and write it to OUTPUT."""
        out = _run_inference(checkpoint, input, output, threshold)
        _typer.echo(f"Wrote georeferenced mask -> {out}")


def main() -> None:
    """Console entry point."""
    if app is None:  # pragma: no cover
        raise SystemExit("typer is required for the inference CLI (pixi install).")
    app()


if __name__ == "__main__":
    main()
