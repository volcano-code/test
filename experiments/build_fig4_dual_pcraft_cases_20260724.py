#!/usr/bin/env python3
"""Build two input-only audited Figure 4 cases for P-CRAFT qualification."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT))

from experiments.build_fixed3_mr1_connected10_pending_20260722 import (
    S4_2D,
    S6_3D,
    component_count,
    iou,
    load_mask,
    save_mask,
    visual_hull,
)


OUTPUT = ROOT / "results/fig4_dual_pcraft_priority_20260724"

CASES = {
    "anime_future_warrior_magic_scholar_commander": [
        Path("/home/lin/xvxinyao/shadow/anime_shadow/005_未来战士的剪影.png"),
        Path("/home/lin/xvxinyao/shadow/anime_shadow/019_魔法学者的黑色剪影.png"),
        Path("/home/lin/xvxinyao/shadow/anime_shadow/06_英勇的指挥官轮廓.png"),
    ],
    "anime_catgirl_ninja_prince": [
        ROOT / "results/distinct_anime_silhouette_screen_20260709/selected/045_076_cat_ear_girl.png",
        ROOT / "results/distinct_anime_silhouette_screen_20260709/selected/032_12_神秘忍者轮廓.png",
        ROOT / "results/distinct_anime_silhouette_screen_20260709/selected/008_18_小王子剪影.png",
    ],
    "normal_beetle_parrot_eagle": [
        Path("/home/lin/xvxinyao/shadow/normal_shadow/bugs_beetle.png"),
        Path("/home/lin/xvxinyao/shadow/normal_shadow/birds_parrot.png"),
        Path("/home/lin/xvxinyao/shadow/normal_shadow/birds_eagle.png"),
    ],
}


def audit(paths: list[Path]) -> dict:
    masks = [load_mask(path, 256) for path in paths]
    hull = visual_hull(masks)
    projections = [hull.any(axis=axis) for axis in range(3)]
    view_ious = [iou(prediction, target) for prediction, target in zip(projections, masks, strict=True)]
    return {
        "resolution": 256,
        "occupied_voxels": int(hull.sum()),
        "components_6": component_count(hull, S6_3D),
        "view_ious": view_ious,
        "mean_iou": float(np.mean(view_ious)),
        "min_iou": float(np.min(view_ious)),
    }


def contact_sheet(cases: list[dict], destination: Path) -> None:
    tile = 256
    label_width = 260
    canvas = Image.new("RGB", (label_width + 3 * tile, len(cases) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, case in enumerate(cases):
        y = row * tile
        draw.text((16, y + 90), case["case_id"], fill="black", font=font)
        draw.text((16, y + 112), "X / Y / Z, original masks", fill=(0, 110, 90), font=font)
        for column, path_text in enumerate(case["target_paths_xyz"]):
            image = Image.open(path_text).convert("RGB")
            canvas.paste(image, (label_width + column * tile, y))
    canvas.save(destination)


def main() -> None:
    inputs = OUTPUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, sources in CASES.items():
        targets = []
        input_audit = []
        for axis, source in zip(("x", "y", "z"), sources, strict=True):
            mask = load_mask(source, 256)
            target = inputs / f"{case_id}_view_{axis}.png"
            save_mask(mask, target)
            targets.append(str(target.resolve()))
            input_audit.append({
                "axis": axis,
                "source_path": str(source.resolve()),
                "canonical_path": str(target.resolve()),
                "foreground_fraction": float(mask.mean()),
                "components_4": component_count(mask, S4_2D),
                "mirrored_components_4": component_count(np.fliplr(mask), S4_2D),
            })
        cases.append({
            "case_id": case_id,
            "stratum": "fig4-input-only-priority",
            "target_paths_xyz": targets,
            "sadekar_tfs_targets": [targets[1], targets[0], targets[2]],
            "input_connectivity_audit": input_audit,
            "visual_hull_audit": audit(sources),
        })

    payload = {
        "schema": "fig4_dual_pcraft_priority_v1",
        "status": "PCRAFT_QUALIFICATION",
        "purpose": "Run P-CRAFT before any Figure 4 baseline evaluation",
        "selection_policy": "Input-only selection; no baseline or P-CRAFT output used.",
        "protocol": {
            "views": 3,
            "declared_rendered_views": 6,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS",
            "tsf": 1,
            "mr": 1,
            "zdist": 1.7,
            "resolution": 256,
            "seeds": [17],
            "evaluation": "shared canonical masks and prescribed mirrors; no registration",
        },
        "cases": cases,
    }
    manifest = OUTPUT / "benchmark_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    contact_sheet(cases, OUTPUT / "input_contact_sheet.png")
    print(json.dumps({
        "manifest": str(manifest),
        "contact_sheet": str(OUTPUT / "input_contact_sheet.png"),
        "cases": [
            {
                "case_id": case["case_id"],
                "hull_components": case["visual_hull_audit"]["components_6"],
                "hull_mean_iou": case["visual_hull_audit"]["mean_iou"],
                "hull_min_iou": case["visual_hull_audit"]["min_iou"],
            }
            for case in cases
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
