"""Command-line entry point for the log-anomaly pipeline.

Three subcommands map onto the pipeline stages:

    loganomaly parse    --config config/hdfs.yaml --out outputs
    loganomaly detect   --config config/hdfs.yaml --out outputs
    loganomaly evaluate --config config/hdfs.yaml --out outputs

``parse`` masks + templates the raw logs into a per-session event-count matrix,
``detect`` scores each session with the configured detector (pca / zscore), and
``evaluate`` compares the flags against the HDFS labels.

The heavy imports (typer for the app, pyspark / sklearn for scale-out parsing)
happen inside the functions, so importing this module never requires the full
stack. The core numeric work uses the pure-numpy modules in this package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_parse(config_path: str | Path, out_dir: str | Path) -> Path:
    """Parse raw logs into a session event-count Parquet (Spark)."""
    from loganomaly.spark_pipeline import parse_logs_to_counts

    cfg = _load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return parse_logs_to_counts(
        cfg["input"]["glob"],
        cfg["input"]["session_regex"],
        out / "event_counts.parquet",
    )


def run_detect(config_path: str | Path, out_dir: str | Path) -> Path:
    """Score the event-count matrix and write per-session flags."""
    import numpy as np
    import pandas as pd

    from loganomaly import detect as d

    cfg = _load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(out / "event_counts.parquet")
    pivot = df.pivot_table(
        index="session", columns="template", values="count", fill_value=0
    ).sort_index()
    X = pivot.to_numpy(dtype=float)

    dcfg = cfg["detect"]
    if dcfg["method"] == "pca":
        errors = d.pca_reconstruction_error(X, dcfg["pca"]["k"])
        flags = d.flag(errors, dcfg["pca"]["quantile"])
    elif dcfg["method"] == "zscore":
        errors = X.sum(axis=1)
        flags = d.zscore_anomalies(errors, dcfg["zscore"]["z"])
    else:
        raise ValueError(f"Unknown detector: {dcfg['method']!r}")

    result = pd.DataFrame({"session": pivot.index, "score": errors, "flag": flags})
    path = out / "flags.parquet"
    result.to_parquet(path)
    print(
        json.dumps(
            {"n_sessions": int(len(result)), "n_flagged": int(np.sum(flags))}, indent=2
        )
    )
    return path


def run_evaluate(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Score the flags against the HDFS labels (precision / recall / F1)."""
    import pandas as pd

    from loganomaly.evaluate import confusion_matrix, precision_recall_f1

    cfg = _load_config(config_path)
    out = Path(out_dir)

    flags = pd.read_parquet(out / "flags.parquet").set_index("session")
    ecfg = cfg["evaluate"]
    labels = pd.read_csv(ecfg["label_path"])
    y_true_map = (
        labels.set_index(ecfg["block_column"])[ecfg["label_column"]]
        == ecfg["anomaly_value"]
    )

    joined = flags.join(y_true_map.rename("y_true"), how="inner")
    y_true = joined["y_true"].to_numpy(dtype=bool)
    y_pred = joined["flag"].to_numpy(dtype=bool)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    summary = {
        "n": int(len(joined)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    """Build the typer app and dispatch. Imports typer lazily."""
    import typer

    app = typer.Typer(add_completion=False, help=__doc__)

    @app.command()
    def parse(config: str = "config/hdfs.yaml", out: str = "outputs") -> None:
        """Mask + template raw logs into event-count features."""
        run_parse(config, out)

    @app.command()
    def detect(config: str = "config/hdfs.yaml", out: str = "outputs") -> None:
        """Score sessions with the configured detector."""
        run_detect(config, out)

    @app.command()
    def evaluate(config: str = "config/hdfs.yaml", out: str = "outputs") -> None:
        """Compare flags against HDFS labels."""
        run_evaluate(config, out)

    @app.command()
    def demo(seed: int = 0, out: str = "outputs") -> None:
        """Run the end-to-end demo on a seeded synthetic log (no Spark)."""
        from loganomaly.demo import run_demo

        print(json.dumps(run_demo(seed, out), indent=2))

    app(args=argv)


if __name__ == "__main__":  # pragma: no cover
    main()
