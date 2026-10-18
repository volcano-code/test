#!/usr/bin/env python3
"""Freeze a complex fixed-three-view benchmark using input-only criteria.

The manifest deliberately contains no method output or method-conditioned
selection.  It is therefore suitable for a common-frame quantitative study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
COMPLEX = ROOT / "data/silhouettes/complex_v2_filled"
ANIME = Path("/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified")


# One object-style silhouette and two independently selected character-style
# silhouettes per task.  Files were chosen before method execution to cover
# thin appendages, holes, concavities, and articulated outlines.
CASES = [
    ("complex01", "complex_phoenix.png", "A012_012_优雅的伞下武士.png", "A016_016_黑暗骑士与长矛.png"),
    ("complex02", "complex_scifi_city.png", "A105_anime30_26.png", "A013_013_狐女神秘身影.png"),
    ("complex03", "complex_stag.png", "A071_22_优雅舞者的剪影.png", "A092_anime30_13.png"),
    ("complex04", "complex_dragon.png", "A029_027_哥特风格的小提琴手.png", "A030_028_狐耳巫女的舞动身影.png"),
    ("complex05", "complex_ship.png", "A051_11_勇敢战士的动态轮廓.png", "A079_30_机巧少女与小机器人.png"),
    ("complex06", "complex_cathedral.png", "A106_anime30_27.png", "A075_26_优雅的狐女剪影.png"),
    ("complex07", "complex_bicycle.png", "A024_022_未来机械战士的身影.png", "A002_002_魔法少女剪影.png"),
    ("complex08", "complex_castle.png", "A095_anime30_16.png", "A027_025_黑暗骑士的孤独身影.png"),
    ("complex09", "complex_robot.png", "A034_030_自信少女与棒球棒.png", "A010_010_未来街头英雄轮廓.png"),
    ("complex10", "complex_bonsai.png", "A053_12_活力四射的机械少女.png", "A098_anime30_19.png"),
    ("complex11", "complex_owl_totem.png", "A101_anime30_22.png", "A107_anime30_28.png"),
    ("complex12", "complex_violin.png", "A006_006_活泼的猫耳少女剪影.png", "A073_24_滑板空中动作.png"),
]


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((256, 256), Image.Resampling.NEAREST)
    array = np.asarray(image)
    # All benchmark assets use white foreground on black.  The fallback makes
    # the convention explicit if an accidentally inverted asset is supplied.
    mask = array >= 128
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def mask_metrics(path: Path) -> dict[str, float | int]:
    mask = load_mask(path)
    structure = ndimage.generate_binary_structure(2, 2)
    _, components = ndimage.label(mask, structure)
    filled = ndimage.binary_fill_holes(mask)
    _, holes = ndimage.label(filled & ~mask, structure)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    boundary = mask & ~eroded
    area = int(mask.sum())
    return {
        "foreground_fraction": float(mask.mean()),
        "components_8": int(components),
        "holes_8": int(holes),
        "boundary_pixels": int(boundary.sum()),
        "boundary_over_sqrt_area": float(boundary.sum() / np.sqrt(max(area, 1))),
    }


def resolve(name: str, index: int) -> Path:
    path = (COMPLEX if index == 0 else ANIME) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/complex_fixed3_input_frozen_20260721/benchmark_manifest.json",
    )
    args = parser.parse_args()

    rows = []
    for case_id, *names in CASES:
        paths = [resolve(name, index) for index, name in enumerate(names)]
        metrics = [mask_metrics(path) for path in paths]
        rows.append(
            {
                "case_id": case_id,
                "stratum": "input-complex",
                "target_paths_xyz": [str(path) for path in paths],
                # Sadekar TFS consumes Y, X, Z while the common evaluator uses
                # declared X, Y, Z.  This is a protocol mapping, not registration.
                "sadekar_tfs_targets": [str(paths[1]), str(paths[0]), str(paths[2])],
                "input_only_selection": {
                    "per_view": metrics,
                    "mean_boundary_over_sqrt_area": float(
                        np.mean([row["boundary_over_sqrt_area"] for row in metrics])
                    ),
                    "total_holes_8": int(sum(row["holes_8"] for row in metrics)),
                },
            }
        )

    payload = {
        "schema": "complex_fixed3_input_frozen_v1",
        "purpose": "primary common-frame comparison on input-selected complex silhouettes",
        "selection_policy": (
            "Frozen before method execution using silhouette-only shape criteria; "
            "no method output, IoU, mesh status, or candidate eligibility was used."
        ),
        "protocol": {
            "views": 3,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS, mirror_mode=false, zdist=1.7",
            "resolution": 256,
            "seeds": [17, 29, 43],
            "evaluation": "shared original masks and common fixed-origin renderer; no registration, reorientation, or rescaling",
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
