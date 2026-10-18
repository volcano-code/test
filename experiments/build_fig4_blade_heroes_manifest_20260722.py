#!/usr/bin/env python3
"""Freeze the single three-view anime case used for Figure 4 evaluation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
INPUT = ROOT / "results/table2_anime_fixed3_inputs_20260722/blade_heroes"
OUTPUT = ROOT / "results/fig4_blade_heroes_fixed3_mr1_20260722/benchmark_manifest.json"


def main() -> None:
    xyz = [str((INPUT / f"view_{index}.png").resolve()) for index in range(3)]
    missing = [path for path in xyz if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Figure 4 target masks: {missing}")
    payload = {
        "schema": "fig4_blade_heroes_fixed3_mr1_v1",
        "purpose": "single-case qualitative comparison; frozen before baseline execution",
        "selection_policy": (
            "The case was selected for visible thin weapons, capes, ribbons, and limb gaps. "
            "It is not used to compute the aggregate Table 1 benchmark."
        ),
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
            "evaluation": "shared masks and prescribed mirror views; no per-method alignment",
        },
        "cases": [{
            "case_id": "blade_heroes",
            "stratum": "anime-thin-structure-qualitative",
            "target_paths_xyz": xyz,
            "sadekar_tfs_targets": [xyz[1], xyz[0], xyz[2]],
            "input_only_selection": {
                "difficulty_score": 1.0,
                "rationale": "thin weapons, capes, ribbons, and separated limb boundaries",
                "per_view": [{"path": path} for path in xyz],
            },
        }],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
