#!/usr/bin/env python3
"""Build the fixed-camera three-view benchmark used for author-code baselines.

The manifest is intentionally separate from the two-wall physical benchmark.
It freezes the three Sadekar TFS cameras and the target ordering before any
Sadekar output is evaluated.  No per-model centering, scale fitting, or
orientation search is allowed by this protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def input_for(root: Path, index: int) -> Path:
    paths = sorted((root / "inputs").glob(f"input{index}_*.png"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one input{index} image in {root}, found {paths}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/nsa_style_fixed_required_10cm_256/MANIFEST.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/controlled_fixed3_author_tsf_20260716",
    )
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text())
    rows = []
    for group in source["cases"]:
        for model in group["models"]:
            order = model.get("physical_order")
            if not order:
                continue
            root = Path(model["stl"]).parent
            # Sadekar's non-mirrored TFS camera order is front, side, top.
            # The physical assets are stored as top(z), side(x), front(y).
            inputs = {
                "top_z": input_for(root, 0),
                "side_x": input_for(root, 1),
                "front_y": input_for(root, 2),
            }
            rows.append({
                "case_id": model["example"],
                "source_group": group["case_id"],
                "stratification_role": "pending",
                "physical_targets": {
                    "top_z": str(inputs["top_z"]),
                    "side_x": str(inputs["side_x"]),
                    "front_y": str(inputs["front_y"]),
                },
                "sadekar_tfs_targets": [
                    str(inputs["front_y"]), str(inputs["side_x"]), str(inputs["top_z"]),
                ],
                "shadowforge_voxels256": model["voxels"],
                "shadowforge_stl": model["stl"],
            })

    strata_path = ROOT / "results/mitra_pauly2009_fixed3view_20260716/mitra_difficulty_strata.json"
    if strata_path.is_file():
        strata = json.loads(strata_path.read_text())
        by_case = {
            row["case_id"]: row["stratum"]
            for key in ("low_mitra_cases", "high_mitra_cases")
            for row in strata[key]
        }
        for row in rows:
            row["stratification_role"] = by_case.get(row["case_id"], "unassigned")

    payload = {
        "schema": "shadowforge_fixed3_author_tsf_v1",
        "purpose": "same-camera baseline reconstruction; primary aggregate uses all 12 rows",
        "protocol": {
            "views": 3,
            "camera": "Sadekar create_cameras_TFS_mode, orthographic, mirror_mode=false",
            "camera_order": ["front_y", "side_x", "top_z"],
            "world_extent": 1.7,
            "voxel_resolution": 256,
            "mesh_mapping": "fixed world extent and origin; independent mesh rescaling is forbidden",
            "score_renderer": "frozen common TFS orthographic renderer",
            "sadekar_native_run": {"mirror_mode": False, "tsf_mode": True, "zdist": 1.7},
            "seeds": [17, 29, 43],
        },
        "cases": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "benchmark_manifest.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(destination), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
