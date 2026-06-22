"""Command-line interface for clinicaccess (numpy/pandas only).

Two subcommands:

* ``demo``   -- run the seeded synthetic demo and write artifacts.
* ``report`` -- run the nearest-facility + coverage pipeline on your own
  places/facilities CSVs and print the headline numbers.

Run with ``python -m clinicaccess.cli demo`` or, once installed, ``clinicaccess
demo``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinicaccess.access import coverage_stats, farthest_places, nearest_facility
from clinicaccess.dataio import load_facilities, load_places
from clinicaccess.demo import THRESHOLDS_KM, run_demo


def _cmd_demo(args: argparse.Namespace) -> int:
    m = run_demo(seed=args.seed, out_dir=args.out)
    print("clinic-access demo (seeded synthetic Cameroon points, straight-line distance)")
    print(f"  places={m['n_places']}  facilities={m['n_facilities']}")
    print(f"  mean nearest   = {m['mean_nearest_km']:.1f} km")
    print(f"  median nearest = {m['median_nearest_km']:.1f} km")
    print(f"  share within 5 km  = {m['share_within_5km']:.1%}")
    print(f"  share within 10 km = {m['share_within_10km']:.1%}")
    print(f"  share beyond 25 km = {m['share_beyond_25km']:.1%}")
    print(f"  farthest place = {m['farthest_place_km']:.1f} km")
    print(f"  artifacts -> {args.out}/places_access.csv, {args.out}/summary.json")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Run the access pipeline on real CSVs (lat/lon required; population optional)."""
    places = load_places(args.places)
    facilities = load_facilities(args.facilities)
    if "population" not in places.columns:
        places = places.assign(population=1.0)

    access = nearest_facility(places, facilities)
    stats = coverage_stats(access["nearest_km"], access["population"], args.thresholds)
    farthest = farthest_places(access, n=args.farthest_n)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    access.to_csv(out_dir / "places_access.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_places": int(len(places)),
                "n_facilities": int(len(facilities)),
                "thresholds_km": list(args.thresholds),
                "mean_nearest_km": float(access["nearest_km"].mean()),
                "median_nearest_km": float(access["nearest_km"].median()),
                "coverage": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"clinic-access report: {len(places)} places, {len(facilities)} facilities")
    print(f"  mean nearest   = {access['nearest_km'].mean():.1f} km")
    print(f"  median nearest = {access['nearest_km'].median():.1f} km")
    print(f"  farthest {args.farthest_n} (most underserved):")
    for _, row in farthest.iterrows():
        label = row.get("name", row.name)
        print(f"    {label}: {row['nearest_km']:.1f} km")
    print(f"  artifacts -> {out_dir}/places_access.csv, {out_dir}/summary.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clinicaccess",
        description="Straight-line clinic-access screening (Cameroon).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run the seeded synthetic demo.")
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--out", type=Path, default="outputs")
    p_demo.set_defaults(func=_cmd_demo)

    p_report = sub.add_parser("report", help="Run the access pipeline on your own CSVs.")
    p_report.add_argument("places", type=Path, help="Places CSV (lat, lon, population).")
    p_report.add_argument("facilities", type=Path, help="Facilities CSV (lat, lon).")
    p_report.add_argument("--out", type=Path, default="outputs")
    p_report.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=THRESHOLDS_KM,
        help="Distance thresholds in km (default: 5 10 25).",
    )
    p_report.add_argument("--farthest-n", type=int, default=10, dest="farthest_n")
    p_report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
