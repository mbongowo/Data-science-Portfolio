"""Command-line entry point for the sentiment-scale pipeline.

Three subcommands mirror the three stages:

    sentiment ingest --config config/reddit.yaml --out data/raw
    sentiment score  --config config/reddit.yaml --data data/raw --out outputs
    sentiment trends --config config/reddit.yaml --data outputs --out outputs

``ingest`` and ``score`` reach into Spark; ``trends`` is pure pandas. The heavy
imports (typer, pyspark, the lexicon model) all happen inside the command
functions, so importing this module never requires the full stack and the pure
core is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_paths(cfg: dict[str, Any], raw_dir: str | Path) -> list[str]:
    """Resolve the per-subreddit dump paths under ``raw_dir`` from the config."""
    raw = Path(raw_dir)
    field = "comments"
    return [str(raw / f"{sub}_{field}.zst") for sub in cfg["corpus"]["subreddits"]]


def ingest(config: str, out: str) -> None:
    """Load compressed Reddit dumps, clean them, and write Parquet."""
    from sentiment import spark_nlp

    cfg = _load_config(config)
    spark = spark_nlp.get_spark()
    df = spark_nlp.load_dumps(
        spark, _dump_paths(cfg, out), text_field=cfg["corpus"]["text_field"]
    )
    spark_nlp.write_parquet(df, str(Path(out) / "documents.parquet"))


def score(config: str, data: str, out: str) -> None:
    """Score the ingested documents with the configured scorer."""
    from sentiment import spark_nlp

    cfg = _load_config(config)
    scfg = cfg["scoring"]
    spark = spark_nlp.get_spark()
    df = spark.read.parquet(str(Path(data) / "documents.parquet"))

    if scfg["scorer"] == "model":
        scored = spark_nlp.score_model_spark(
            df, batch_size=scfg["model"]["batch_size"]
        )
    else:
        lex = scfg["lexicon"]
        scored = spark_nlp.score_lexicon_spark(
            df,
            negation_window=lex["negation_window"],
            alpha=lex["alpha"],
        )
    spark_nlp.write_parquet(scored, str(Path(out) / "scored.parquet"))


def trends(config: str, data: str, out: str) -> None:
    """Aggregate scored documents into a daily/weekly sentiment series."""
    import pandas as pd

    from sentiment.aggregate import sentiment_timeseries

    cfg = _load_config(config)
    df = pd.read_parquet(str(Path(data) / "scored.parquet"))
    series = sentiment_timeseries(df, freq=cfg["aggregation"]["freq"])

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(out_dir / "sentiment_timeseries.csv", index=False)
    summary = {
        "freq": cfg["aggregation"]["freq"],
        "n_periods": int(len(series)),
        "n_documents": int(series["n"].sum()),
        "mean_score_overall": float(df["score"].mean()),
    }
    with open(out_dir / "trends_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def _build_app() -> Any:
    import typer

    app = typer.Typer(add_completion=False, help=__doc__)
    app.command()(ingest)
    app.command()(score)
    app.command()(trends)
    return app


def main(argv: list[str] | None = None) -> int:
    """Entry point declared in ``[project.scripts]``."""
    app = _build_app()
    app(args=argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
