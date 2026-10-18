#!/usr/bin/env python3
"""Package 12 representative 10 cm shadow-art print examples.

The package only copies models with existing manifest evidence. It does not
relax quality gates or edit meshes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/nsa_style_shadowart_refs_12_10cm_256"


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def item_from_symbolic(example: str, panel: str, case_id: str) -> dict:
    data = load("results/curated_symbolic_fixed_10cm_256/MANIFEST.json")
    row = next(r for r in data["examples"] if r["example"] == example)
    if not row.get("accepted"):
        raise RuntimeError(f"{example} is not accepted")
    selected = row["selected"]
    return {
        "case_id": case_id,
        "nsa_panel": panel,
        "example": example,
        "silhouettes": row["silhouettes"],
        "source_manifest": str(ROOT / "results/curated_symbolic_fixed_10cm_256/MANIFEST.json"),
        "source_kind": "rebuilt_symbolic_fixed",
        "selected_branch": "fixed",
        "candidate": selected["candidate"],
        "iou": selected["iou"],
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
    out["source_manifest"] = str(ROOT / "results/nsa_style_curated_12_10cm_256/BATCH_MANIFEST.json")
    out["source_kind"] = "previous_curated_pass"
    out["src_dir"] = str(Path(row["stl"]).parent)
    return out


ITEMS = [
    item_from_symbolic("fig1_A_heart", "top-left", "case1_corner_clear_plate"),
    item_from_symbolic("fig1_A_gear", "top-left", "case1_corner_clear_plate"),
    item_from_complex("complex_C_cathedral_bicycle_owl", "top-middle", "case2_corner_suspended_object"),
    item_from_complex("complex_F_ship_phoenix_robot", "top-middle", "case2_corner_suspended_object"),
    item_from_symbolic("complex_robot_dragon_violin", "top-right", "case3_single_wall_like"),
    item_from_curated("case3A_spider_scorpion_gear", "top-right", "case3_single_wall_like"),
    item_from_symbolic("fig1_I_tree", "bottom-left", "case4_white_paper_sweep"),
    item_from_curated("case3B_lobster_squid_ladybug", "bottom-left", "case4_white_paper_sweep"),
    item_from_curated("case4A_ant_peacock_gear", "bottom-middle", "case5_acrylic_center_display"),
    item_from_curated("case4B_shrimp_turtle_owl", "bottom-middle", "case5_acrylic_center_display"),
    item_from_curated("case6A_bat_trex_snail", "bottom-right", "case6_plinth_like_display"),
    item_from_curated("case6B_ladybug_frog_peacock", "bottom-right", "case6_plinth_like_display"),
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rows = []
    for idx, item in enumerate(ITEMS, 1):
        dst_dir = OUT / item["case_id"] / item["example"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("inputs", "silhouettes"):
            src_sub = Path(item["src_dir"]) / sub
            if src_sub.exists():
                shutil.copytree(src_sub, dst_dir / sub)

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
        "schema": "nsa_style_shadowart_refs_12_10cm_256_package_v1",
        "basis": "Representative package following NSA/Sadekar-style use of complex objects, characters/symbols, and recognizable silhouettes. Every row is copied from a manifest-backed generated model; no mesh is manually edited.",
        "resolution": 256,
        "print_mm": 100.0,
        "quality_gate": {
            "iou_min": 0.975,
            "saving_range": [0.40, 0.85],
            "single": True,
            "watertight": True,
        },
        "aaai_A_note": "aaai_A was regenerated with internal hole preservation before rebuilding symbolic models.",
        "n_models": len(rows),
        "models": rows,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "BATCH_MANIFEST.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE -> {OUT}")
    for row in rows:
        print(f"{row['package_index']:02d} {row['example']} IoU={row['iou']} saving={row['saving']} stl={row['stl']}")


if __name__ == "__main__":
    main()
