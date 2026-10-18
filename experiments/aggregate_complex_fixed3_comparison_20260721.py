#!/usr/bin/env python3
"""Aggregate the input-frozen complex benchmark under one metric protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


def stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def summarize(name: str, rows: list[dict]) -> dict:
    mean_iou, std_iou = stats([float(row["mean_iou"]) for row in rows])
    worst_mean, worst_std = stats([float(row["worst_iou"]) for row in rows])
    return {
        "method": name,
        "tasks": len(rows),
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "mean_worst_view_iou": worst_mean,
        "std_worst_view_iou": worst_std,
        "single_body_rate": float(np.mean([bool(row["single_body"]) for row in rows])),
        "watertight_rate": float(np.mean([bool(row["watertight"]) for row in rows])),
    }


def mesh_watertight(path: Path) -> bool:
    if not path.is_file():
        raise RuntimeError(f"Missing mesh for watertight QA: {path}")
    loaded = trimesh.load(path, process=False, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            return False
        return bool(all(mesh.is_watertight for mesh in geometries))
    return bool(loaded.is_watertight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mitra", type=Path, required=True, help="Directory containing one metrics.json per case")
    parser.add_argument("--sadekar", type=Path, required=True, help="Common-renderer unified_metrics.json")
    parser.add_argument("--shadowforge", type=Path, required=True, help="ShadowForge SUMMARY.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = [row["case_id"] for row in manifest["cases"]]

    mitra_rows = []
    for case_id in expected:
        path = args.mitra / case_id / "metrics.json"
        if not path.is_file():
            raise RuntimeError(f"Missing Mitra metrics: {path}")
        row = json.loads(path.read_text())
        mitra_rows.append({
            "mean_iou": row["mean_iou"],
            "worst_iou": row["worst_iou"],
            "single_body": row["single_body"],
            "watertight": row.get(
                "watertight",
                mesh_watertight(args.mitra / case_id / "mitra_pauly2009_shadow_hull.obj"),
            ),
        })

    sad_payload = json.loads(args.sadekar.read_text())
    sad_aggregate = sad_payload.get("aggregate_against_original_targets") or sad_payload["aggregate"]
    sad_rows_by_method: dict[str, list[dict]] = {}
    for method in sad_aggregate:
        per_case = method.get("per_case_seed_averages", [])
        found = {row["case_id"] for row in per_case}
        missing = sorted(set(expected) - found)
        if missing:
            raise RuntimeError(f"Missing {method['method']} task metrics: {missing}")
        sad_rows_by_method[method["method"]] = [
            {
                "mean_iou": row["mean_iou"],
                "worst_iou": row["min_view_iou"],
                "single_body": row["single_body"] >= 0.5,
                "watertight": row["watertight"] >= 0.5,
            }
            for row in per_case if row["case_id"] in expected
        ]

    sf_payload = json.loads(args.shadowforge.read_text())
    sf_by_case = {row["case_id"]: row for row in sf_payload if row.get("case_id") in expected}
    missing_sf = sorted(set(expected) - set(sf_by_case))
    if missing_sf:
        raise RuntimeError(f"Missing ShadowForge tasks: {missing_sf}")
    sf_rows = []
    for case_id in expected:
        row = sf_by_case[case_id]
        selected = row.get("selected") or {}
        mean_iou = row.get("mean_iou", selected.get("mean_iou"))
        worst_iou = row.get("min_iou", selected.get("min_iou"))
        components = row.get("components6", selected.get("components"))
        watertight = row.get("watertight", selected.get("watertight"))
        if None in (mean_iou, worst_iou, components, watertight):
            raise RuntimeError(f"Incomplete ShadowForge delivery for {case_id}")
        sf_rows.append({
            "mean_iou": mean_iou,
            "worst_iou": worst_iou,
            "single_body": int(components) == 1,
            "watertight": bool(watertight),
        })

    aggregate = [summarize("Mitra-style", mitra_rows)]
    for method in ("Sadekar voxel", "Sadekar mesh"):
        if method not in sad_rows_by_method:
            raise RuntimeError(f"Missing Sadekar aggregate: {method}")
        aggregate.append(summarize(method, sad_rows_by_method[method]))
    aggregate.append(summarize("P-CRAFT", sf_rows))

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "selection_policy": manifest["selection_policy"],
        "protocol": manifest["protocol"],
        "aggregate": aggregate,
        "note": "Mean and worst-view statistics are population mean/std across task-level values; Sadekar seeds are averaged within task first.",
    }
    (args.output / "table1_real.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "table1_real.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregate)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
