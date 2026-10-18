#!/usr/bin/env python3
"""Package NSA-style 10 cm print examples with corrected Fig.1 n=2 models.

The old package included Fig.1-like symbolic models rebuilt with three input
silhouettes.  This package replaces them with the current paper Fig.1 overview
models, each of which has exactly two silhouettes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/nsa_style_shadowart_refs_12_with_fig1_n2_10cm_256"


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def item_from_fig1_n2(example: str, panel: str, case_id: str) -> dict:
    data = load("results/fig1_overview_n2_10cm_256/MANIFEST.json")
    row = next(r for r in data["examples"] if r["example"] == example)
    if row["n_inputs"] != 2:
        raise RuntimeError(f"{example} is not n=2: {row['n_inputs']}")
    if not (row["components_6conn"] == 1 and row["watertight"]):
        raise RuntimeError(f"{example} is not printable by mesh/connectivity evidence: {row}")
    return {
        "case_id": case_id,
        "nsa_panel": panel,
        "example": example,
        "silhouettes": row["targets"],
        "n_inputs": row["n_inputs"],
        "views": row["views"],
        "source_manifest": str(ROOT / "results/fig1_overview_n2_10cm_256/MANIFEST.json"),
        "source_kind": "current_fig1_overview_n2",
        "selected_branch": "visual_hull_n2",
        "candidate": "two_view_visual_hull",
        "iou": row["iou_mean"],
        "view_ious": row["view_ious"],
        "saving": None,
        "saving_note": "Not comparable to three-view material-saving gate; this row is included as the current Fig.1 n=2 physical-print asset.",
        "single": row["components_6conn"] == 1,
        "watertight": row["watertight"],
        "erosion_survival": row["erosion_survival"],
        "mesh_extents_mm": row["mesh_extents_mm"],
        "mesh_faces": row["mesh_faces"],
        "stl": row["stl"],
        "voxels": row["voxels"],
        "src_dir": str(Path(row["stl"]).parent),
    }


def item_from_complex(example: str, panel: str, case_id: str) -> dict:
    data = load("results/complex_v2_free_fixed_10cm_256/MANIFEST.json")
    row = next(r for r in data["examples"] if r["example"] == example)
    selected = row["selected"]
    if not (selected["mean_iou"] >= 0.975 and 0.40 <= selected["saving"] <= 0.85 and selected["single"] and selected["watertight"]):
        raise RuntimeError(f"{example} selected candidate violates the package gate: {selected}")
    return {
        "case_id": case_id,
        "nsa_panel": panel,
        "example": example,
        "silhouettes": row["silhouettes"],
        "n_inputs": len(row["silhouettes"]),
        "source_manifest": str(ROOT / "results/complex_v2_free_fixed_10cm_256/MANIFEST.json"),
        "source_kind": "complex_free_fixed",
        "selected_branch": selected["branch"],
        "candidate": selected["candidate"],
        "iou": selected["mean_iou"],
        "view_ious": selected["view_ious"],
        "saving": selected["saving"],
        "single": selected["single"],
        "watertight": selected["watertight"],
        "mesh_extents_mm": selected["mesh_extents_mm"],
        "mesh_faces": selected["mesh_faces"],
        "stl": selected["stl"],
        "voxels": selected["voxels"],
        "src_dir": str(Path(selected["stl"]).parent),
    }


def item_from_curated(example: str, panel: str, case_id: str) -> dict:
    rows = load("results/nsa_style_curated_12_10cm_256/BATCH_MANIFEST.json")
    row = next(r for r in rows if r["example"] == example)
    if not (row["iou"] >= 0.975 and 0.40 <= row["saving"] <= 0.85 and row["single"] and row["watertight"]):
        raise RuntimeError(f"{example} violates the package gate: {row}")
    out = dict(row)
    out["case_id"] = case_id
    out["nsa_panel"] = panel
    out["n_inputs"] = len(row["silhouettes"])
    out["source_manifest"] = str(ROOT / "results/nsa_style_curated_12_10cm_256/BATCH_MANIFEST.json")
    out["source_kind"] = "previous_curated_pass"
    out["src_dir"] = str(Path(row["stl"]).parent)
    return out


ITEMS = [
    item_from_fig1_n2("model_A1_spade", "top-left", "case1_fig1_corner_two_view"),
    item_from_fig1_n2("model_A2_house", "top-left", "case1_fig1_corner_two_view"),
    item_from_fig1_n2("model_A3_teddy", "top-middle", "case2_fig1_corner_two_view"),
    item_from_fig1_n2("model_I_sword", "top-middle", "case2_fig1_corner_two_view"),
    item_from_complex("complex_C_cathedral_bicycle_owl", "top-right", "case3_single_wall_like"),
    item_from_complex("complex_F_ship_phoenix_robot", "top-right", "case3_single_wall_like"),
    item_from_curated("case3A_spider_scorpion_gear", "bottom-left", "case4_white_paper_sweep"),
    item_from_curated("case3B_lobster_squid_ladybug", "bottom-left", "case4_white_paper_sweep"),
    item_from_curated("case4A_ant_peacock_gear", "bottom-middle", "case5_acrylic_center_display"),
    item_from_curated("case4B_shrimp_turtle_owl", "bottom-middle", "case5_acrylic_center_display"),
    item_from_curated("case6A_bat_trex_snail", "bottom-right", "case6_plinth_like_display"),
    item_from_curated("case6B_ladybug_frog_peacock", "bottom-right", "case6_plinth_like_display"),
]


def copy_optional_tree(src: Path, dst: Path) -> None:
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []
    for idx, item in enumerate(ITEMS, 1):
        dst_dir = OUT / item["case_id"] / item["example"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_dir = Path(item["src_dir"])
        copy_optional_tree(src_dir / "inputs", dst_dir / "inputs")
        copy_optional_tree(src_dir / "silhouettes", dst_dir / "silhouettes")
        copy_optional_tree(src_dir / "projections", dst_dir / "projections")

        stl_src = Path(item["stl"])
        npy_src = Path(item["voxels"])
        stl_dst = dst_dir / stl_src.name
        npy_dst = dst_dir / npy_src.name
        shutil.copy2(stl_src, stl_dst)
        shutil.copy2(npy_src, npy_dst)

        rec = {k: v for k, v in item.items() if k != "src_dir"}
        rec["package_index"] = idx
        rec["stl"] = str(stl_dst)
        rec["voxels"] = str(npy_dst)
        rows.append(rec)

    summary = {
        "schema": "nsa_style_shadowart_refs_12_with_fig1_n2_10cm_256_v1",
        "basis": "NSA-style physical print package with the current paper Fig.1 n=2 models integrated. The previous three-input Fig.1-like symbolic rows are intentionally excluded.",
        "resolution": 256,
        "print_mm": 100.0,
        "n_models": len(rows),
        "fig1_n2_models": [r["example"] for r in rows if r["source_kind"] == "current_fig1_overview_n2"],
        "quality_gate_for_three_view_rows": {
            "iou_min": 0.975,
            "saving_range": [0.40, 0.85],
            "single": True,
            "watertight": True,
        },
        "fig1_n2_gate": {
            "n_inputs": 2,
            "single": True,
            "watertight": True,
            "print_scale_mm": 100.0,
        },
        "models": rows,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "BATCH_MANIFEST.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE -> {OUT}")
    for row in rows:
        print(f"{row['package_index']:02d} {row['example']} n={row['n_inputs']} IoU={row['iou']} saving={row['saving']} stl={row['stl']}")


if __name__ == "__main__":
    main()
