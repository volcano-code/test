#!/usr/bin/env python3
"""Freeze the geometry-only, fixed-camera benchmark used by the paper."""

from __future__ import annotations

import hashlib
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "results/nsa_style_fixed_required_10cm_256/MANIFEST.json"
OUT = ROOT / "results/controlled_benchmark_geometry_only_20260710"

SPLITS = {
    "validation": [
        "case4A_parrot_frog_peacock",
        "case5A_ant_peacock_frog",
    ],
    "test": [
        "case1A_owl_bat_turtle",
        "case2A_parrot_dragon_snail",
        "case3A_spider_scorpion_beetle",
        "case6B_ladybug_frog_peacock",
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return None


def git_status(path: Path) -> list[str]:
    try:
        return subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], cwd=path, text=True
        ).splitlines()
    except Exception:
        return []


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    by_name = {
        model["example"]: model
        for group in payload["cases"]
        for model in group["models"]
    }
    rows = []
    for split, names in SPLITS.items():
        for name in names:
            src = by_name[name]
            case_dir = Path(src["voxels"]).parent
            inputs = sorted((case_dir / "inputs").glob("input*.png"))
            if len(inputs) != 3:
                raise RuntimeError(f"{name}: expected three inputs, found {len(inputs)}")
            if src.get("selected_branch") != "fixed":
                raise RuntimeError(f"{name}: not a fixed-branch artifact")
            files = [*inputs, Path(src["voxels"]), Path(src["stl"])]
            for path in files:
                if not path.exists():
                    raise FileNotFoundError(path)
            rows.append({
                "case_id": name,
                "split": split,
                "target_names": src["silhouettes"],
                "input_paths": [str(p) for p in inputs],
                "shadowforge_voxels256": src["voxels"],
                "shadowforge_stl100mm": src["stl"],
                "shadowforge_candidate": src["candidate"],
                "shadowforge_source": "geometry",
                "fixed_camera": True,
                "checksums": {str(p): sha256(p) for p in files},
            })

    manifest = {
        "schema": "shadowforge_controlled_fixed_benchmark_v1",
        "source_manifest": str(SOURCE),
        "methods": [
            "Mitra-style carving",
            "Sadekar voxel-based",
            "Sadekar mesh-based",
            "ShadowForge",
        ],
        "protocol": {
            "view_assignment": ["view0->z", "view1->x", "view2->y"],
            "d4_search": False,
            "view_permutation": False,
            "camera_optimization": False,
            "light_optimization": False,
            "target_pixels": [256, 256],
            "evaluation_grid": [256, 256, 256],
            "physical_bbox_mm": 100.0,
            "projector": "hard axis-occupancy projector after coordinate normalization",
            "single_body": "one 6-connected occupied component",
            "watertight": "trimesh.is_watertight on exported/evaluated mesh",
            "saving": "1 - occupied(V256) / occupied(H256)",
            "sadekar_seeds": [17, 29, 43],
            "sadekar_voxel": {"iterations": 600, "lr": 0.01, "threshold": 0.05},
            "sadekar_mesh": {"iterations": 2000, "lr": 0.15},
            "seed_reporting": "mean+-std on test; median-IoU seed displayed",
        },
        "source_revisions": {
            "shadowart_revisited": git_revision(Path("/data/xvxinyao/ShadowArt-Revisited")),
            "shadowart_revisited_tracked_changes": git_status(Path("/data/xvxinyao/ShadowArt-Revisited")),
            "shadowart_revisited_adaptations": [
                "voxel grid changed from 128 to the shared 256 evaluation resolution",
                "boolean CLI parsing corrected so fixed TFS/mirror flags are unambiguous",
                "saved-log IoU/Dice calculation corrected to operate on binary masks; training losses unchanged",
            ],
            "mitra_reimplementation": str(ROOT / "experiments/mitra_baseline.py"),
        },
        "cases": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "benchmark_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (OUT / "benchmark_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "split", "target_names", "input_paths", "shadowforge_voxels256",
            "shadowforge_stl100mm", "shadowforge_candidate", "shadowforge_source", "fixed_camera",
        ])
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["target_names"] = "|".join(row["target_names"])
            flat["input_paths"] = "|".join(row["input_paths"])
            writer.writerow({key: flat[key] for key in writer.fieldnames})
    print(path)


if __name__ == "__main__":
    main()
