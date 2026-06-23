"""Command-line interface for the air-quality streaming demo and stream stubs.

The ``demo`` command drives the pure-Python core (windowing, EPA AQI, alert
engine) over a seeded synthetic stream and is the CI-tested, runnable
contribution. The ``produce`` / ``process`` commands are thin stubs over the
guarded streaming layer; their heavy imports (``argparse`` aside) live inside
:mod:`aqstream.stream` / :mod:`aqstream.ingest`, so importing this module is
cheap and the tested core stays free of optional dependencies.

Run ``python -m aqstream.cli demo`` to reproduce the result numbers.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the seeded synthetic demo end-to-end and print the summary."""
    from aqstream.demo import run_demo

    metrics = run_demo(seed=args.seed, out_dir=args.out_dir)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


def _cmd_produce(args: argparse.Namespace) -> int:
    """Fetch readings from Open-Meteo and publish them to Kafka (lazy)."""
    from aqstream.ingest import fetch_open_meteo_aq, parse_aq
    from aqstream.stream import produce

    payload = fetch_open_meteo_aq(args.lat, args.lon, hours=args.hours)
    readings = parse_aq(payload, station=args.station)
    n = produce(readings, topic=args.topic, bootstrap_servers=args.brokers)
    print(f"produced {n} readings to topic {args.topic!r}")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    """Run the Spark Structured Streaming processor (lazy)."""
    from aqstream.stream import process_stream

    process_stream(args.topic, bootstrap_servers=args.brokers)
    print(f"processing topic {args.topic!r} (Ctrl-C to stop)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI."""
    parser = argparse.ArgumentParser(
        prog="aqstream",
        description="Real-time air-quality streaming & alerting (Cameroon cities).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the seeded synthetic demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out-dir", default="outputs")
    demo.set_defaults(func=_cmd_demo)

    produce = sub.add_parser(
        "produce",
        help="fetch Open-Meteo readings and publish to Kafka (needs the stack)",
    )
    produce.add_argument("--lat", type=float, required=True)
    produce.add_argument("--lon", type=float, required=True)
    produce.add_argument("--station", required=True)
    produce.add_argument("--hours", type=int, default=72)
    produce.add_argument("--topic", default="aqstream.readings")
    produce.add_argument("--brokers", default="localhost:9092")
    produce.set_defaults(func=_cmd_produce)

    process = sub.add_parser(
        "process", help="run the Spark streaming processor (needs the stack)"
    )
    process.add_argument("--topic", default="aqstream.readings")
    process.add_argument("--brokers", default="localhost:9092")
    process.set_defaults(func=_cmd_process)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``aqstream``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
