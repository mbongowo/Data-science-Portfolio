"""Command-line interface for the offline clickstream aggregations.

This exposes the pure-Python / pandas core (windowing, sessionisation, funnel,
events-per-minute) over a CSV of events, so the analysis can be reproduced from
a file without standing up Kafka or Spark. Heavy imports (``typer``, ``pandas``)
are kept inside :func:`main` / the command bodies so that importing this module
is cheap and the package's tested core stays free of optional dependencies.

The expected input CSV has at least the columns ``ts`` (epoch seconds), ``user``
(an identifier), and ``event`` (the event name). Run ``clickstream --help`` for
the available commands once the package is installed.
"""

from __future__ import annotations

from typing import Any


def _load_events(path: str) -> Any:
    """Read an events CSV into a DataFrame (lazy pandas import)."""
    import pandas as pd

    df = pd.read_csv(path)
    missing = {"ts", "user", "event"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def build_app() -> Any:
    """Construct and return the Typer application (lazy ``typer`` import)."""
    import typer

    app = typer.Typer(
        add_completion=False,
        help="Offline clickstream aggregations over a CSV of events.",
    )

    @app.command("per-minute")
    def per_minute(csv: str) -> None:
        """Print event counts per one-minute bucket."""
        from clickstream.aggregate import events_per_minute

        df = _load_events(csv)
        series = events_per_minute(df)
        for minute, count in series.items():
            typer.echo(f"{int(minute)}\t{int(count)}")

    @app.command("tumbling")
    def tumbling(csv: str, window_s: float = 60.0) -> None:
        """Print tumbling-window event counts."""
        from clickstream.windows import tumbling_counts

        df = _load_events(csv)
        events = list(zip(df["ts"].tolist(), df["event"].tolist(), strict=True))
        for start, count in sorted(tumbling_counts(events, window_s).items()):
            typer.echo(f"{start}\t{count}")

    @app.command("sessions")
    def sessions(csv: str, user: str, gap_s: float = 1800.0) -> None:
        """Print session ids for one user's events, in time order."""
        from clickstream.windows import sessionize

        df = _load_events(csv)
        sub = df[df["user"].astype(str) == user].sort_values("ts")
        ids = sessionize(sub["ts"].tolist(), gap_s)
        for ts, sid in zip(sub["ts"].tolist(), ids, strict=True):
            typer.echo(f"{ts}\t{sid}")

    @app.command("funnel")
    def funnel_cmd(csv: str, steps: str) -> None:
        """Print funnel reach counts. STEPS is a comma-separated step list."""
        from clickstream.windows import funnel

        df = _load_events(csv).sort_values(["user", "ts"])
        step_list = [s.strip() for s in steps.split(",") if s.strip()]
        user_events: dict[Any, list[str]] = {}
        for user_id, group in df.groupby("user"):
            user_events[user_id] = group["event"].tolist()
        counts = funnel(user_events, step_list)
        for name, count in zip(step_list, counts, strict=True):
            typer.echo(f"{name}\t{count}")

    @app.command("demo")
    def demo(seed: int = 0, out_dir: str = "outputs") -> None:
        """Run the seeded synthetic demo end-to-end and print the summary."""
        import json

        from clickstream.demo import run_demo

        metrics = run_demo(seed=seed, out_dir=out_dir)
        typer.echo(json.dumps(metrics, indent=2, default=str))

    return app


def main() -> None:
    """Console-script entry point (``clickstream``)."""
    app = build_app()
    app()


if __name__ == "__main__":
    main()
