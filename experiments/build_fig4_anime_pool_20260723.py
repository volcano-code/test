#!/usr/bin/env python3
"""Build a fixed, pre-registered three-view anime pool for Fig. 4 evaluation.

The pool is frozen before any method is run.  It is deliberately separate from
the Table 1 benchmark: a qualitative figure may only be chosen from eligible
P-CRAFT deliveries, using the selection rule recorded in the manifest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = Path("/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified")
OUT = ROOT / "results/fig4_anime_pool_fixed3_mr1_20260723"


CASES = {
    "regal_blade": [
        "A016_016_黑暗骑士与长矛.png",
        "A019_019_魔法学者的黑色剪影.png",
        "A028_026_英姿飒爽的少女武者.png",
    ],
    "umbrella_adventure": [
        "A012_012_优雅的伞下武士.png",
        "A037_04_年轻冒险者的自信身影.png",
        "A019_019_魔法学者的黑色剪影.png",
    ],
    "warrior_regal": [
        "A028_026_英姿飒爽的少女武者.png",
        "A016_016_黑暗骑士与长矛.png",
        "A037_04_年轻冒险者的自信身影.png",
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, names in CASES.items():
        case_dir = OUT / "inputs" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for view, name in enumerate(names):
            src = SOURCE / name
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = case_dir / f"view_{view}.png"
            shutil.copy2(src, dst)
            paths.append(str(dst))
        # Sadekar's TFS convention consumes Y, X, Z in its existing launcher.
        cases.append({
            "case_id": case_id,
            "source_files": names,
            "target_paths_xyz": paths,
            "sadekar_tfs_targets": [paths[1], paths[0], paths[2]],
        })

    manifest = {
        "title": "Frozen Fig. 4 anime pool, fixed three-view mirrored protocol",
        "protocol": {
            "views": 3,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS",
            "resolution": 256,
            "mr": 1,
            "tsf": 1,
            "zdist": 1.7,
            "seeds": [17],
        },
        "selection_policy": {
            "pool_is_frozen_before_method_runs": True,
            "figure_candidate_eligibility": "P-CRAFT must be eligible, single-body, and watertight.",
            "ranking": "maximize P-CRAFT mean IoU minus the strongest baseline mean IoU; break ties by P-CRAFT worst-view IoU.",
            "scope": "This qualitative choice is not used for aggregate Table 1 reporting.",
        },
        "cases": cases,
    }
    path = OUT / "benchmark_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(path), "cases": [row["case_id"] for row in cases]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
