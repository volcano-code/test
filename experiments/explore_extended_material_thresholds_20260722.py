#!/usr/bin/env python3
"""Reuse frozen soft fields to probe lighter P-CRAFT delivery candidates.

This experiment changes only the threshold ensemble used by deterministic
repair, guarded carving, and delivery. It never retrains the soft proposal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")


def input_paths(manifest: dict) -> list[str]:
    metrics = manifest.get("soft", {}).get("metrics", {})
    paths = metrics.get("args", {}).get("shadow_files")
    if not paths:
        paths = metrics.get("config", {}).get("silhouettes")
    if not paths:
        paths = manifest.get("soft", {}).get("metrics", {}).get("input_paths")
    if not paths:
        paths = manifest.get("input_paths")
    if not paths:
        raise KeyError("Could not recover silhouette paths from source manifest")
    return [str(Path(path).resolve()) for path in paths[:3]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Existing fixed3 P-CRAFT result root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--thresholds", default="0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifests = sorted(source.glob("connected_*/seed_*/MANIFEST.json"))
    rows: list[dict] = []
    for manifest_path in source_manifests:
        case_id = manifest_path.parents[1].name
        seed = int(manifest_path.parent.name.removeprefix("seed_"))
        if args.case and case_id not in args.case:
            continue
        if args.seed and seed not in args.seed:
            continue
        run = output / case_id / f"seed_{seed}"
        target_manifest = run / "MANIFEST.json"
        if target_manifest.is_file() and not args.force:
            payload = json.loads(target_manifest.read_text())
            rows.append({"case_id": case_id, "seed": seed, "status": "reused",
                         "selected": payload.get("selected")})
            continue

        source_payload = json.loads(manifest_path.read_text())
        silhouettes = input_paths(source_payload)
        artifact_dir = manifest_path.parent / "soft_training"
        run.mkdir(parents=True, exist_ok=True)
        command = [
            str(PY), str(ROOT / "shadowforge_pipeline/unified_run.py"),
            "--silhouettes", *silhouettes,
            "--output", str(run), "--branch", "soft-geometry",
            "--tsf", "1", "--mr", "1",
            "--soft-resolution", "256", "--soft-control-res", "0",
            "--soft-proposal-mode", "proposal",
            "--soft-artifact-dir", str(artifact_dir.resolve()),
            "--core-resolution", "48",
            "--soft-seed-thresholds", args.thresholds,
            "--bridge-radius", "2", "--bridge-passes", "48",
            "--delivery-shell-voxels", "2",
            "--min-iou", "0.95", "--min-view-iou", "0.90",
            "--min-saving", "0.000001", "--max-saving", "0.85",
            "--selection-iou-tolerance", "0.01",
            "--selection-policy", "material", "--screen", "skip",
            "--device", "cuda:0",
        ]
        with (run / "RUN_OUTPUT.txt").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        payload = json.loads(target_manifest.read_text()) if target_manifest.is_file() else {}
        selected = payload.get("selected") or {}
        row = {
            "case_id": case_id,
            "seed": seed,
            "returncode": completed.returncode,
            "status": payload.get("status", "ERROR"),
            "mean_iou": selected.get("mean_iou"),
            "min_iou": selected.get("min_iou"),
            "saving": selected.get("saving"),
            "candidate": selected.get("candidate"),
        }
        rows.append(row)
        (output / "SUMMARY.partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    (output / "SUMMARY.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
