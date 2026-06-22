"""Real TorchGeo ResNet18 fine-tune on EuroSAT (needs a GPU / Colab).

Everything in this module is **heavy and lazy**: torch / torchvision / torchgeo
are imported *inside* the functions, never at module load, so importing
``lcnet.train`` (for ``--help`` or in a test) does not require the deep-learning
stack and does not pull a GPU into CI. The pure-numpy demo in :mod:`lcnet.demo`
covers the reproducible, CI-tested path; this module is the documented route to
the actual model result that drops into the README's model card.

Run it on a machine with a CUDA GPU (or a Colab GPU runtime) after installing the
full stack (``pip install -r requirements.txt`` including torch / torchgeo). The
EuroSAT dataset (Sentinel-2 land-cover patches) downloads automatically through
TorchGeo on first use.

Credit: the dataset loader, the ImageNet-pretrained weights, and the geospatial
sampling utilities come from microsoft/torchgeo
(https://github.com/microsoft/torchgeo).
"""

from __future__ import annotations

from typing import Any

__all__ = ["train_eurosat", "compare_transfer"]


def train_eurosat(
    pretrained: bool,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_classes: int = 10,
    seed: int = 0,
    data_root: str = "data/eurosat",
    device: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a ResNet18 on EuroSAT and return held-out metrics.

    Parameters
    ----------
    pretrained : bool
        If ``True`` start from ImageNet weights and fine-tune; if ``False`` train
        the same architecture from random initialisation. This single flag is the
        whole pretrained-vs-scratch comparison.
    epochs, batch_size, lr : int, int, float
        Standard training hyper-parameters.
    num_classes : int
        EuroSAT has 10 land-cover classes.
    seed : int
        Seed for torch / numpy / python RNGs.
    data_root : str
        Where TorchGeo downloads / caches EuroSAT.
    device : str, optional
        ``"cuda"`` / ``"cpu"``; auto-detected when ``None``.

    Returns
    -------
    dict
        ``{"pretrained", "epochs", "test_accuracy", "test_macro_f1",
        "confusion_matrix"}``. The confusion matrix and macro-F1 are computed
        with this repo's pure-numpy :mod:`lcnet.metrics`, so the same metric code
        scores both the baseline and the deep model.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch.utils.data import DataLoader, random_split  # noqa: PLC0415
    from torchgeo.datasets import EuroSAT  # noqa: PLC0415
    from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415

    from lcnet.metrics import confusion_matrix, macro_f1  # noqa: PLC0415

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # EuroSAT via TorchGeo. The RGB bands keep this compatible with the ImageNet
    # ResNet stem; the multispectral variant would need a wider first conv.
    dataset = EuroSAT(root=data_root, download=True, bands=EuroSAT.rgb_bands)
    n_test = int(0.2 * len(dataset))
    n_train = len(dataset) - n_test
    gen = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(dev)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in train_dl:
            images = batch["image"].float().to(dev)
            labels = batch["label"].to(dev)
            optim.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optim.step()

    # Evaluate, scoring with this repo's pure-numpy metrics.
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in test_dl:
            images = batch["image"].float().to(dev)
            preds = model(images).argmax(dim=1).cpu().numpy()
            y_pred.extend(int(p) for p in preds)
            y_true.extend(int(t) for t in batch["label"].numpy())

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    cm = confusion_matrix(y_true_arr, y_pred_arr, num_classes)
    acc = float(np.mean(y_true_arr == y_pred_arr)) if y_true_arr.size else 0.0
    return {
        "pretrained": bool(pretrained),
        "epochs": int(epochs),
        "test_accuracy": acc,
        "test_macro_f1": macro_f1(y_true_arr, y_pred_arr, num_classes),
        "confusion_matrix": cm.tolist(),
    }


def compare_transfer(epochs: int = 10, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Run pretrained vs from-scratch fine-tunes and return both metric dicts.

    This is the transfer-learning comparison framework: the same architecture,
    data split, and metric code, differing only in whether ImageNet weights seed
    the run. Fill the README's model-card table from the returned dict.

    Returns
    -------
    dict
        ``{"pretrained": <metrics>, "scratch": <metrics>}``.
    """
    return {
        "pretrained": train_eurosat(pretrained=True, epochs=epochs, **kwargs),
        "scratch": train_eurosat(pretrained=False, epochs=epochs, **kwargs),
    }
