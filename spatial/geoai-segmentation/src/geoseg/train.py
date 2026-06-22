"""Hydra training entry point.

Composes the Hydra config (``conf/config.yaml`` -> model + data), seeds
everything, instantiates the datamodule + model + Lightning Trainer, and logs
the *resolved* config and the current git SHA to MLflow so every run is
traceable back to exact code + settings.

Run::

    python -m geoseg.train                 # full run
    python -m geoseg.train trainer.fast_dev_run=true   # 1-step smoke
"""

from __future__ import annotations

import subprocess
from typing import Any

__all__ = ["git_sha", "main"]


def git_sha(default: str = "unknown") -> str:
    """Return the current git commit SHA, or ``default`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return default


def _run(cfg: Any) -> dict:
    """Core training routine (kept separate so it is unit-testable)."""
    import mlflow  # noqa: PLC0415
    import pytorch_lightning as pl  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from pytorch_lightning.loggers import MLFlowLogger  # noqa: PLC0415

    from geoseg.datamodule import GeoSegDataModule
    from geoseg.model import build_model
    from geoseg.seed import seed_everything

    seed_everything(cfg.seed, deterministic=cfg.get("deterministic", True))

    datamodule = GeoSegDataModule(
        data_dir=cfg.data.data_dir,
        tile_size=cfg.data.tile_size,
        stride=cfg.data.get("stride", None),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        band_means=cfg.data.get("band_means", None),
        band_stds=cfg.data.get("band_stds", None),
        seed=cfg.seed,
    )
    model = build_model(cfg.model)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.get("experiment_name", "geoseg"),
        tracking_uri=cfg.get("mlflow_tracking_uri", "file:./mlruns"),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.get("accelerator", "auto"),
        devices=cfg.trainer.get("devices", "auto"),
        fast_dev_run=cfg.trainer.get("fast_dev_run", False),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        default_root_dir=cfg.paths.get("output_dir", "./outputs"),
        logger=mlflow_logger,
    )

    # Log provenance: resolved config + git SHA.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_tracking_uri(mlflow_logger._tracking_uri)
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_dict(resolved, "resolved_config.yaml")

    trainer.fit(model, datamodule=datamodule)
    return {"git_sha": git_sha(), "run_id": mlflow_logger.run_id}


def main() -> None:
    """Hydra-decorated console entry point."""
    import hydra  # noqa: PLC0415

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def _main(cfg) -> None:
        from omegaconf import OmegaConf  # noqa: PLC0415

        print(OmegaConf.to_yaml(cfg))
        _run(cfg)

    _main()


if __name__ == "__main__":
    main()
