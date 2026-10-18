#!/usr/bin/env python3
"""Aggregate all methods on the frozen fixed3/mr1 challenge benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh


def mean_std(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()), float(data.std(ddof=0))


def summarize(method: str, rows: list[dict]) -> dict:
    iou, iou_std = mean_std([row["mean_iou"] for row in rows])
    worst, worst_std = mean_std([row["worst_iou"] for row in rows])
    return {
        "method": method,
        "tasks": len(rows),
        "mean_iou": iou,
        "std_iou": iou_std,
        "mean_worst_view_iou": worst,
        "std_worst_view_iou": worst_std,
        "single_body_rate": float(np.mean([row["single_body"] for row in rows])),
        "watertight_rate": float(np.mean([row["watertight"] for row in rows])),
        "delivery_rate": float(np.mean([row["single_body"] and row["watertight"] for row in rows])),
    }


def mesh_watertight(path: Path) -> bool:
    mesh = trimesh.load(path, process=True, force="mesh")
    return bool(mesh.is_watertight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mitra", type=Path, required=True)
    parser.add_argument("--sadekar", type=Path, required=True)
    parser.add_argument("--shadowforge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.manifest.read_text())
    expected = [row["case_id"] for row in benchmark["cases"]]

    mitra_rows = []
    for case_id in expected:
        case_dir = args.mitra / case_id
        data = json.loads((case_dir / "metrics.json").read_text())
        mesh_path = case_dir / "mitra_pauly2009_shadow_hull.obj"
        mitra_rows.append({
            "case_id": case_id, "mean_iou": float(data["mean_iou"]),
            "worst_iou": float(data["worst_iou"]), "single_body": bool(data["single_body"]),
            "watertight": (
                mesh_watertight(mesh_path)
                if mesh_path.is_file()
                else bool(data.get("digital_watertight", False))
            ),
        })

    sad = json.loads(args.sadekar.read_text())
    sad_original = sad["aggregate_against_original_targets"]
    sad_rows: dict[str, list[dict]] = {}
    for method in sad_original:
        by_case = {row["case_id"]: row for row in method["per_case_seed_averages"]}
        missing = sorted(set(expected) - set(by_case))
        if missing:
            raise RuntimeError(f"Missing {method['method']} cases: {missing}")
        sad_rows[method["method"]] = [{
            "case_id": case_id, "mean_iou": float(by_case[case_id]["mean_iou"]),
            "worst_iou": float(by_case[case_id]["min_view_iou"]),
            "single_body": bool(by_case[case_id]["single_body"] >= 0.5),
            "watertight": bool(by_case[case_id]["watertight"] >= 0.5),
        } for case_id in expected]

    sf_summary = json.loads((args.shadowforge / "SUMMARY.json").read_text())
    grouped: dict[str, list[dict]] = {case_id: [] for case_id in expected}
    for run in sf_summary:
        if run.get("case_id") in grouped and run.get("selected"):
            grouped[run["case_id"]].append(run)
    sf_rows = []
    for case_id in expected:
        runs = grouped[case_id]
        if not runs:
            raise RuntimeError(f"No ShadowForge result for {case_id}")
        selected = [run["selected"] for run in runs]
        sf_rows.append({
            "case_id": case_id,
            "mean_iou": float(np.mean([row["mean_iou"] for row in selected])),
            "worst_iou": float(np.mean([row["min_iou"] for row in selected])),
            "single_body": bool(np.mean([int(row["components"] == 1) for row in selected]) >= 0.5),
            "watertight": bool(np.mean([int(row["watertight"]) for row in selected]) >= 0.5),
            "eligible": bool(np.mean([int(row.get("eligible", False)) for row in selected]) >= 0.5),
            "saving": float(np.mean([row["saving"] for row in selected])),
        })

    per_case = []
    sources = {
        "Mitra-style": mitra_rows,
        "Sadekar voxel": sad_rows["Sadekar voxel"],
        "Sadekar mesh": sad_rows["Sadekar mesh"],
        "P-CRAFT": sf_rows,
    }
    for method, rows in sources.items():
        per_case.extend({"method": method, **row} for row in rows)
    aggregate = [summarize(method, rows) for method, rows in sources.items()]
    aggregate[-1]["eligible_rate"] = float(np.mean([row["eligible"] for row in sf_rows]))
    aggregate[-1]["mean_saving"] = float(np.mean([row["saving"] for row in sf_rows]))

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete", "protocol": benchmark["protocol"],
        "selection_policy": benchmark["selection_policy"], "aggregate": aggregate,
        "per_case": per_case,
        "note": "Sadekar is scored against shared original masks and mirrored counterparts; no author-only target normalization is used.",
    }
    (args.output / "table1_real.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, rows in (("table1_real.tsv", aggregate), ("table1_per_case.tsv", per_case)):
        keys = sorted({key for row in rows for key in row})
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
