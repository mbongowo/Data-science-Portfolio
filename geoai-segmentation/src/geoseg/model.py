"""LightningModule wrapping an SMP U-Net for binary segmentation.

Loss is a combination of Dice + BCE-with-logits. Train/val loss and IoU are
logged each epoch (MLflow picks these up via the configured logger).

Heavy imports (torch, lightning, segmentation_models_pytorch) are performed
lazily so that ``import geoseg.model`` works without them; the LightningModule
class is only assembled when first instantiated.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_model", "SegmentationModule"]


def _import_stack():
    import pytorch_lightning as pl  # noqa: PLC0415
    import segmentation_models_pytorch as smp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    return pl, smp, torch, nn


class SegmentationModule:
    """Lightning wrapper around ``smp.Unet``.

    Rebases onto ``pytorch_lightning.LightningModule`` at instantiation time so
    the module imports cleanly without lightning installed.
    """

    def __new__(cls, *args, **kwargs):
        try:
            import pytorch_lightning as pl  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs lightning
            raise ImportError(
                "SegmentationModule requires pytorch_lightning. Run "
                "`pixi install` to provision the full environment."
            ) from exc
        if pl.LightningModule not in cls.__bases__:
            cls.__bases__ = (pl.LightningModule, *cls.__bases__)
        return super().__new__(cls)

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        lr: float = 1e-3,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        pl, smp, torch, nn = _import_stack()

        self.lr = lr
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    # --- forward / loss -------------------------------------------------
    def forward(self, x):  # pragma: no cover - needs torch
        return self.net(x)

    def _compute_loss(self, logits, target):  # pragma: no cover - needs torch
        return (
            self.dice_weight * self.dice_loss(logits, target)
            + self.bce_weight * self.bce_loss(logits, target)
        )

    def _iou(self, logits, target):  # pragma: no cover - needs torch
        import torch  # noqa: PLC0415

        preds = (torch.sigmoid(logits) > 0.5).float()
        target = (target > 0.5).float()
        inter = (preds * target).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
        iou = (inter + 1e-7) / (union + 1e-7)
        return iou.mean()

    # --- lightning steps ------------------------------------------------
    def _step(self, batch, stage: str):  # pragma: no cover - needs torch
        logits = self(batch["image"])
        loss = self._compute_loss(logits, batch["mask"])
        iou = self._iou(logits, batch["mask"])
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):  # pragma: no cover - needs torch
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):  # pragma: no cover - needs torch
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):  # pragma: no cover - needs torch
        return self._step(batch, "test")

    def configure_optimizers(self):  # pragma: no cover - needs torch
        import torch  # noqa: PLC0415

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_model(cfg: Any) -> "SegmentationModule":
    """Instantiate :class:`SegmentationModule` from a config object/mapping."""
    get = cfg.get if hasattr(cfg, "get") else lambda k, d=None: getattr(cfg, k, d)
    return SegmentationModule(
        encoder_name=get("encoder_name", "resnet34"),
        encoder_weights=get("encoder_weights", "imagenet"),
        in_channels=get("in_channels", 3),
        classes=get("classes", 1),
        lr=get("lr", 1e-3),
    )
