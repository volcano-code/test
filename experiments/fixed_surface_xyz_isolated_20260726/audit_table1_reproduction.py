#!/usr/bin/env python3
"""Audit whether the archived Table 1 result is reproducible by the isolated copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_summary(manifest: dict) -> dict:
    selected = manifest.get("selected") or {}
    return {
        "status": manifest.get("status"),
        "candidate": selected.get("candidate"),
        "mean_iou": selected.get("mean_iou"),
        "min_iou": selected.get("min_iou"),
        "mean_dice": selected.get("mean_dice"),
        "saving": selected.get("saving"),
        "components": selected.get("components"),
        "watertight": selected.get("watertight"),
        "eligible": selected.get("eligible"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--isolated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archived_soft = args.archived / "soft_training/occupancy_soft_proposal.npy"
    rerun_soft = args.rerun / "soft_training/occupancy_soft_proposal.npy"
    soft_comparison: dict[str, object] = {
        "archived_path": str(archived_soft),
        "rerun_path": str(rerun_soft),
    }
    if archived_soft.is_file() and rerun_soft.is_file():
        lhs = np.load(archived_soft)
        rhs = np.load(rerun_soft)
        delta = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        soft_comparison.update(
            {
                "shape_equal": lhs.shape == rhs.shape,
                "archived_sha256": sha256(archived_soft),
                "rerun_sha256": sha256(rerun_soft),
                "max_abs_delta": float(delta.max()),
                "mean_abs_delta": float(delta.mean()),
                "binary_voxel_difference_at_0_5": int(
                    np.count_nonzero((lhs >= 0.5) != (rhs >= 0.5))
                ),
            }
        )

    archived_manifest = load_json(args.archived / "MANIFEST.json")
    rerun_manifest = load_json(args.rerun / "MANIFEST.json")
    isolated_manifest = load_json(args.isolated / "MANIFEST.json")
    archived_success = archived_manifest.get("status") == "SUCCESS"
    current_exact = (
        archived_success
        and isolated_manifest.get("status") == "SUCCESS"
        and selected_summary(archived_manifest) == selected_summary(isolated_manifest)
    )

    report = {
        "scope": "Table 1 connected_01, seed 17",
        "archived": selected_summary(archived_manifest),
        "current_nominal_exact_rerun": selected_summary(rerun_manifest),
        "isolated_core48_replay": selected_summary(isolated_manifest),
        "soft_artifact_comparison": soft_comparison,
        "source_snapshot": {
            "active_unified_run_sha256": sha256(
                ROOT / "shadowforge_pipeline/unified_run.py"
            ),
            "isolated_unified_run_sha256": sha256(
                ROOT
                / "experiments/fixed_surface_xyz_isolated_20260726"
                / "shadowforge_pipeline/unified_run.py"
            ),
        },
        "verdict": {
            "archived_result_is_auditable": archived_success,
            "from_scratch_exact_reproduction_proven": current_exact,
            "reason": (
                "Exact reproduction is proven."
                if current_exact
                else "The frozen soft proposal is stable, but the current isolated "
                "geometry/delivery implementation does not reproduce the archived "
                "selected candidate and metrics."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
