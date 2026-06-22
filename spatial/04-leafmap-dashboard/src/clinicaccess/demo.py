"""Runnable, self-contained demo of the clinic-access screening pipeline.

Synthesizes ~200 populated places and ~20 health facilities scattered across a
Cameroon bounding box with a seeded RNG, runs the real nearest-facility,
coverage and ranking functions on them, and writes artifacts. No network, no
geospatial stack. The points are synthetic and seeded -- reproducible but
illustrative, not measured from real Cameroon data.

Run it with ``python -m clinicaccess.demo`` or ``python -m clinicaccess.cli demo``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinicaccess.access import coverage_stats, farthest_places, nearest_facility

# Cameroon-ish bounding box (W, S, E, N) in WGS84.
CAMEROON_BBOX = (8.5, 2.0, 16.0, 13.0)
THRESHOLDS_KM = [5, 10, 25]
N_PLACES = 200
N_FACILITIES = 20
FARTHEST_N = 10


def _synthesize(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw seeded synthetic places and facilities inside the Cameroon bbox.

    Facilities are scattered across the bbox. Most places cluster near a
    facility (people and clinics both follow towns and roads), but a minority are
    drawn uniformly anywhere in the bbox to stand in for remote, underserved
    settlements. That mix gives a believable distance distribution: a bulk within
    a few tens of km and an underserved tail -- which is what the thresholds and
    the farthest-N ranking are there to surface.
    """
    rng = np.random.default_rng(seed)
    w, s, e, n = CAMEROON_BBOX

    fac_lon = rng.uniform(w, e, size=N_FACILITIES)
    fac_lat = rng.uniform(s, n, size=N_FACILITIES)
    facilities = pd.DataFrame(
        {
            "facility_id": np.arange(N_FACILITIES),
            "name": [f"Facility {i:02d}" for i in range(N_FACILITIES)],
            "lat": fac_lat,
            "lon": fac_lon,
        }
    )

    # ~80% of places sit near a (randomly chosen) facility with a small spread
    # in degrees (~0.15 deg ~= 17 km sigma); the rest are remote outliers.
    n_clustered = int(round(0.8 * N_PLACES))
    n_remote = N_PLACES - n_clustered
    anchors = rng.integers(0, N_FACILITIES, size=n_clustered)
    clustered_lat = fac_lat[anchors] + rng.normal(0.0, 0.15, size=n_clustered)
    clustered_lon = fac_lon[anchors] + rng.normal(0.0, 0.15, size=n_clustered)
    remote_lat = rng.uniform(s, n, size=n_remote)
    remote_lon = rng.uniform(w, e, size=n_remote)

    place_lat = np.clip(np.concatenate([clustered_lat, remote_lat]), s, n)
    place_lon = np.clip(np.concatenate([clustered_lon, remote_lon]), w, e)
    # Lumpy but positive population, rounded to whole people.
    population = rng.gamma(shape=2.0, scale=3500.0, size=N_PLACES).round().astype(int)

    order = rng.permutation(N_PLACES)
    places = pd.DataFrame(
        {
            "place_id": np.arange(N_PLACES),
            "name": [f"Place {i:03d}" for i in range(N_PLACES)],
            "lat": place_lat[order],
            "lon": place_lon[order],
            "population": population,
        }
    )
    return places, facilities


def run_demo(seed: int = 0, out_dir: str | Path = "outputs") -> dict:
    """Run the full demo and write artifacts; return headline metrics.

    Synthesizes ~200 places and ~20 facilities, computes each place's
    straight-line distance to its nearest facility, summarises coverage within
    5/10/25 km, and ranks the 10 farthest (most underserved) places. Writes
    ``places_access.csv`` and ``summary.json`` to ``out_dir``.

    Returns a dict with ``n_places``, ``n_facilities``, ``mean_nearest_km``,
    ``median_nearest_km``, ``share_within_5km``, ``share_within_10km``,
    ``share_beyond_25km`` and ``farthest_place_km``.
    """
    places, facilities = _synthesize(seed)

    access = nearest_facility(places, facilities)
    stats = coverage_stats(access["nearest_km"], access["population"], THRESHOLDS_KM)
    farthest = farthest_places(access, n=FARTHEST_N)

    mean_km = float(access["nearest_km"].mean())
    median_km = float(access["nearest_km"].median())
    farthest_km = float(access["nearest_km"].max())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    access.to_csv(out_dir / "places_access.csv", index=False)

    summary = {
        "seed": seed,
        "bbox": CAMEROON_BBOX,
        "n_places": int(len(places)),
        "n_facilities": int(len(facilities)),
        "thresholds_km": THRESHOLDS_KM,
        "mean_nearest_km": mean_km,
        "median_nearest_km": median_km,
        "coverage": stats,
        "farthest_places": [
            {
                "name": row["name"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "population": int(row["population"]),
                "nearest_km": float(row["nearest_km"]),
            }
            for _, row in farthest.iterrows()
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "n_places": int(len(places)),
        "n_facilities": int(len(facilities)),
        "mean_nearest_km": mean_km,
        "median_nearest_km": median_km,
        "share_within_5km": stats["share_within_5km"],
        "share_within_10km": stats["share_within_10km"],
        "share_beyond_25km": stats["share_beyond_25km"],
        "farthest_place_km": farthest_km,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the seeded clinic-access demo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default="outputs")
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
