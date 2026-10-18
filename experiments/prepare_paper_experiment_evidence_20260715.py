#!/usr/bin/env python3
"""Create a non-destructive evidence manifest for the final paper figures.

This script does not run GPU experiments and does not edit the manuscript.  It
records only existing artifacts plus the controlled protocols that still need
to be executed, so every paper figure can be traced to a concrete source.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/paper_experiment_prep_20260715"
BASELINE_V2 = ROOT / "results/physical_print_controlled_baselines_v2_20260715"

SOURCES = [
    ("dark_knight_magic_scholar", "benchmark", ROOT / "results/dark_knight_fdm_7cm_evidence_20260713"),
    ("sword_fox", "benchmark", ROOT / "results/physical_print_soft_geometry_20260714/H02_sword_fox"),
    ("future_mech", "benchmark", ROOT / "results/geometry_delivery_ablation_pilot_20260714/future_mech/full"),
    ("parrot_snail", "benchmark", ROOT / "results/geometry_delivery_ablation_pilot_20260714/parrot_snail/full"),
    ("archer_heroine", "calibration_only", ROOT / "results/fem_two_wall_print_candidate_search_20260714/anime/H05_archer_heroine"),
    ("street_mecha", "candidate", ROOT / "results/fem_two_wall_print_candidate_search_20260714/anime/H03_street_mecha"),
    ("knight_mage", "candidate", ROOT / "results/fem_two_wall_print_candidate_search_20260714/anime/H06_knight_mage"),
    ("frog_beetle", "candidate", ROOT / "results/fem_two_wall_print_candidate_search_20260714/simple/L03_frog_beetle"),
]


def existing(path: Path | None) -> str:
    return str(path) if path and path.is_file() else ""


def read_manifest(source: Path) -> dict:
    manifest = source / "MANIFEST.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return {}


def first_value(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    selected = data.get("selected", {})
    for key in keys:
        if key in selected:
            return selected[key]
    return ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, role, source in SOURCES:
        data = read_manifest(source)
        inputs = source / "inputs"
        row = {
            "task": name,
            "role": role,
            "source_dir": str(source),
            "target_x": existing(inputs / "left_wall_x_target.png"),
            "target_y": existing(inputs / "right_wall_y_target.png"),
            "soft_probability_256": existing(source / "soft_probability_256.npy"),
            "soft_proposal_mesh": existing(source / "soft_proposal_voxelized_mesh.stl"),
            "selected_volume": existing(source / "selected_display_voxels256.npy"),
            "display_stl": existing(source / "selected_display_fixed.stl"),
            "print_stl": existing(source / "selected_print_display_x_down.stl") or existing(source / "selected_print_display_y_down.stl"),
            "candidate": first_value(data, "candidate"),
            "mean_iou": first_value(data, "mean_iou"),
            "mean_dice": first_value(data, "mean_dice"),
            "saving": first_value(data, "saving"),
            "eligible": first_value(data, "eligible"),
        }
        rows.append(row)

    inventory = OUT / "task_inventory.tsv"
    with inventory.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    fig3 = {
        "figure": "Fig. 3: controlled qualitative baseline comparison",
        "protocol": "Use only the four held-out benchmark rows below. Render all methods with the same target images, binary renderer, resolution, and mesh QA after baseline v2 evaluation completes.",
        "methods": ["Mitra-style carving", "Sadekar mesh", "Sadekar voxel", "ShadowForge"],
        "tasks": [r["task"] for r in rows if r["role"] == "benchmark"],
        "baseline_result_root": str(BASELINE_V2),
        "status": "waiting_for_controlled_baseline_v2",
    }
    (OUT / "fig3_rerender_manifest.json").write_text(json.dumps(fig3, indent=2) + "\n")

    fig4 = {
        "figure": "Fig. 4: soft-to-mesh conversion comparison",
        "purpose": "Isolate the conversion stage: the same frozen soft probability field is converted by direct marching cubes, SDF or morphology-only repair, and the proposed geometry-aware conversion.",
        "controls": ["same soft probability tensor", "same threshold grid", "same target projections", "same 256-pixel binary renderer", "same mesh QA"],
        "metrics": ["mean IoU", "mean Dice", "single-body", "watertight", "material saving"],
        "anchor_tasks": [r["task"] for r in rows if r["soft_probability_256"]][:2],
        "status": "protocol_ready_not_run",
    }
    (OUT / "fig4_conversion_protocol.json").write_text(json.dumps(fig4, indent=2) + "\n")

    fig6_rows = [
        {"task": "dark_knight_magic_scholar", "physical_status": "user_confirmed_printed", "photo_path": "", "observed_shadow_left": "", "observed_shadow_right": "", "print_scale_mm": "", "filament": "", "layer_height_mm": "", "support_settings": ""},
        {"task": "sword_fox", "physical_status": "pending", "photo_path": "", "observed_shadow_left": "", "observed_shadow_right": "", "print_scale_mm": "", "filament": "", "layer_height_mm": "", "support_settings": ""},
        {"task": "future_mech", "physical_status": "pending", "photo_path": "", "observed_shadow_left": "", "observed_shadow_right": "", "print_scale_mm": "", "filament": "", "layer_height_mm": "", "support_settings": ""},
    ]
    with (OUT / "fig6_physical_realization_template.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fig6_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(fig6_rows)

    readme = """# Paper Experiment Evidence Preparation\n\nThis folder is a non-destructive audit trail prepared on 2026-07-15.\n\n- `task_inventory.tsv`: existing artifact paths; blank paths mean the artifact was not found and must not be claimed.\n- `fig3_rerender_manifest.json`: the held-out cases and uniform protocol for the controlled baseline figure.\n- `fig4_conversion_protocol.json`: the frozen controls required for the soft-to-mesh conversion study.\n- `fig6_physical_realization_template.tsv`: required real-photo and print-setting fields. Only the dark-knight print is user-confirmed; all other physical rows remain pending.\n\nNo reported metric in these files is a replacement for the controlled baseline v2 results.\n"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps({"out": str(OUT), "tasks": len(rows), "fig3_tasks": len(fig3["tasks"]), "fig4_anchors": fig4["anchor_tasks"]}, indent=2))


if __name__ == "__main__":
    main()
