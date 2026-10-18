#!/usr/bin/env python3
"""Create an NSA-style package whose four Fig.1 rows use the full n=2 rebuild."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC_FIG1 = ROOT / "results/fig1_overview_n2_full_pipeline_10cm_256/MANIFEST.json"
OLD = ROOT / "results/nsa_style_shadowart_refs_12_with_fig1_n2_10cm_256/MANIFEST.json"
OUT = ROOT / "results/nsa_style_shadowart_refs_12_with_fig1_n2_full_pipeline_10cm_256"


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main() -> None:
    fig1 = json.loads(SRC_FIG1.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rows = []
    idx = 1
    panel_map = {
        "model_A1_spade": ("top-left", "case1_fig1_full_pipeline"),
        "model_A2_house": ("top-left", "case1_fig1_full_pipeline"),
        "model_A3_teddy": ("top-middle", "case2_fig1_full_pipeline"),
        "model_I_sword": ("top-middle", "case2_fig1_full_pipeline"),
    }
    for item in fig1["examples"]:
        panel, case_id = panel_map[item["example"]]
        src_dir = Path(item["selected"]["stl"]).parent
        dst_dir = OUT / case_id / item["example"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("inputs", "projections", "candidates"):
            copy_tree_if_exists(src_dir / sub, dst_dir / sub)
        stl_src = Path(item["selected"]["stl"])
        vox_src = Path(item["selected"]["voxels"])
        stl_dst = dst_dir / stl_src.name
        vox_dst = dst_dir / vox_src.name
        shutil.copy2(stl_src, stl_dst)
        shutil.copy2(vox_src, vox_dst)
        rows.append({
            "package_index": idx,
            "case_id": case_id,
            "nsa_panel": panel,
            "example": item["example"],
            "source_kind": "current_fig1_n2_full_pipeline_rebuild",
            "source_manifest": str(SRC_FIG1),
            "n_inputs": item["n_inputs"],
            "silhouettes": item["targets"],
            "views": item["views"],
            "accepted": item["accepted"],
            "candidate": item["selected"]["candidate"],
            "iou": item["selected"]["iou"],
            "view_ious": item["selected"]["view_ious"],
            "saving": item["selected"]["material_saved_vs_hull"],
            "single": item["selected"]["component_count_6"] == 1,
            "watertight": item["selected"]["watertight"],
            "mesh_parts": item["selected"]["mesh_parts"],
            "erosion_survival": item["selected"]["erosion_survival"],
            "stl": str(stl_dst),
            "voxels": str(vox_dst),
        })
        idx += 1

    for old_row in old["models"]:
        if old_row["source_kind"] == "current_fig1_overview_n2":
            continue
        src_dir = Path(old_row["stl"]).parent
        dst_dir = OUT / old_row["case_id"] / old_row["example"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("inputs", "silhouettes", "projections"):
            copy_tree_if_exists(src_dir / sub, dst_dir / sub)
        stl_src = Path(old_row["stl"])
        vox_src = Path(old_row["voxels"])
        stl_dst = dst_dir / stl_src.name
        vox_dst = dst_dir / vox_src.name
        shutil.copy2(stl_src, stl_dst)
        shutil.copy2(vox_src, vox_dst)
        row = dict(old_row)
        row["package_index"] = idx
        row["stl"] = str(stl_dst)
        row["voxels"] = str(vox_dst)
        rows.append(row)
        idx += 1

    summary = {
        "schema": "nsa_style_shadowart_refs_12_with_fig1_n2_full_pipeline_10cm_256_v1",
        "basis": "Full-pipeline rebuild of the current Fig.1 n=2 models integrated with the prior NSA-style package. Strict gate status is preserved per row.",
        "n_models": len(rows),
        "fig1_full_pipeline_manifest": str(SRC_FIG1),
        "strict_fig1_accepted": sum(1 for r in rows if r["source_kind"] == "current_fig1_n2_full_pipeline_rebuild" and r["accepted"]),
        "models": rows,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "BATCH_MANIFEST.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE -> {OUT}")
    for row in rows[:4]:
        print(f"{row['example']} accepted={row['accepted']} saving={row['saving']:.4f} IoU={row['iou']:.4f} mesh_parts={row['mesh_parts']}")


if __name__ == "__main__":
    main()
