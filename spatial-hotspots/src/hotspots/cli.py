"""Command-line entry point for the spatial-hotspots ESDA pipeline.

Run the full analysis from a YAML config:

    hotspots --config config/aoi.yaml --data data/raw/iowa_corn_yield_2023.gpkg \
             --out outputs

The pipeline: load areal units + variable -> build weights -> global Moran's I
-> LISA -> Getis-Ord Gi* -> write a summary and (optionally) cluster maps.

The heavy geospatial imports happen inside :func:`run` so that importing this
module (e.g. for ``--help`` or testing) never requires the full stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_weights(gdf: Any, cfg: dict[str, Any]) -> Any:
    from hotspots import weights as wmod

    wcfg = cfg["weights"]
    kind = wcfg["type"]
    rs = wcfg.get("row_standardize", True)
    on_islands = wcfg.get("on_islands", "raise")

    if kind == "queen":
        return wmod.queen_contiguity(gdf, row_standardize=rs, on_islands=on_islands)
    if kind == "distance_band":
        thr = wcfg.get("distance_band", {}).get("threshold")
        return wmod.distance_band(
            gdf, threshold=thr, row_standardize=rs, on_islands=on_islands
        )
    if kind == "knn":
        k = wcfg.get("knn", {}).get("k", 8)
        return wmod.knn(gdf, k=k, row_standardize=rs)
    raise ValueError(f"Unknown weights type: {kind!r}")


def run(config_path: str | Path, data_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Execute the ESDA pipeline and return a JSON-serialisable summary."""
    import geopandas as gpd

    from hotspots import esda
    from hotspots.weights import diagnostics

    cfg = _load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    value_col = cfg["variable"]["value_column"]
    inf = cfg["inference"]

    gdf = gpd.read_file(data_path)
    if cfg["aoi"].get("crs"):
        gdf = gdf.to_crs(cfg["aoi"]["crs"])
    values = gdf[value_col].to_numpy(dtype=float)

    w = _build_weights(gdf, cfg)
    diag = diagnostics(w).as_dict()

    gm = esda.global_moran(
        values, w, permutations=inf["permutations"], seed=inf["seed"]
    )
    lisa = esda.local_moran(
        values,
        w,
        permutations=inf["permutations"],
        significance=inf["significance"],
        seed=inf["seed"],
    )
    gi = esda.getis_ord_gi_star(
        values,
        w,
        permutations=inf["permutations"],
        significance=inf["significance"],
        seed=inf["seed"],
    )

    gdf["lisa_label"] = lisa.labels
    gdf["gi_label"] = gi.labels
    gdf.to_file(out / "esda_result.gpkg", driver="GPKG")

    summary = {
        "n_units": int(len(gdf)),
        "weights": diag,
        "global_moran": {
            "I": gm.I,
            "expected_I": gm.expected_I,
            "p_sim": gm.p_sim,
            "z_sim": gm.z_sim,
            "permutations": gm.permutations,
        },
        "lisa_counts": {
            label: int((lisa.labels == label).sum())
            for label in ("HH", "LL", "LH", "HL", "ns")
        },
        "gi_counts": {
            label: int((gi.labels == label).sum())
            for label in ("hot", "cold", "ns")
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hotspots", description=__doc__)
    parser.add_argument("--config", default="config/aoi.yaml")
    parser.add_argument("--data", required=True, help="Path to the input GeoPackage.")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args(argv)

    summary = run(args.config, args.data, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
