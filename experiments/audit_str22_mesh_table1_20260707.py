#!/usr/bin/env python3
"""Audit STR22 mesh-based artifacts for Table 1 manufacturing columns."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import trimesh


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_globalconsistent_tfs_20260614/results/aaai2027_paper_20260615/e1_benchmark_final")
OUT = ROOT / "results/table1_str22_mesh_qa_20260707"

SETS = [
    "setA_owl_lobster_ant",
    "setB_lobster_beetle_bat",
    "setC_turtle_scorpion_eagle",
    "setD_dragon_trex_scorpion",
]


def load_mesh(path: Path) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    obj.merge_vertices()
    obj.process()
    return obj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in SETS:
        mesh_path = SRC / case / "sadekar_mesh/model/sample_0_output.obj"
        metrics_path = SRC / case / "sadekar_mesh/metrics.json"
        metrics = json.loads(metrics_path.read_text())
        mesh = load_mesh(mesh_path)
        parts = mesh.split(only_watertight=False)
        watertight = bool(mesh.is_watertight)
        single_body = len(parts) == 1
        eligible = watertight and single_body
        rows.append({
            "case": case,
            "mesh_path": str(mesh_path),
            "mean_iou": metrics["mean_iou"],
            "mean_dice": metrics["mean_dice"],
            "single_body": single_body,
            "watertight": watertight,
            "mesh_components": len(parts),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "mesh_volume": float(mesh.volume) if watertight else "NA",
            "validated_saving": "0.0%" if not eligible else "NEEDS_HULL_VOLUME",
            "eligible_for_validated_saving": eligible,
        })

    with (OUT / "str22_mesh_qa.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cases": len(rows),
        "single_body_all": all(r["single_body"] for r in rows),
        "watertight_all": all(r["watertight"] for r in rows),
        "eligible_all": all(r["eligible_for_validated_saving"] for r in rows),
        "single_body_table_value": str(all(r["single_body"] for r in rows)),
        "watertight_table_value": str(all(r["watertight"] for r in rows)),
        "validated_saving_table_value": "0.0%" if not all(r["eligible_for_validated_saving"] for r in rows) else "NEEDS_HULL_VOLUME",
        "evidence": str(OUT / "str22_mesh_qa.tsv"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
