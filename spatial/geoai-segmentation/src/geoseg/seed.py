"""Seed utility for reproducible runs.

Seeds python's ``random``, numpy, and (if available) torch, and sets
deterministic cuDNN flags. The torch import is guarded so the module is usable
in a torch-free environment (e.g. CI running only the pure metric tests).
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["seed_everything"]


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed all relevant RNGs and return the seed used.

    Parameters
    ----------
    seed:
        Integer seed applied to python, numpy and torch.
    deterministic:
        If True, request deterministic algorithms from torch/cuDNN. This can
        slow training but makes runs bit-for-bit reproducible where supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        # torch is optional; pure-python pipelines remain reproducible.
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Opt into deterministic algorithms where torch provides them.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):  # pragma: no cover - old torch
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    return seed
