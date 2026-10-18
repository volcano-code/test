#!/usr/bin/env python3
"""Prepare three fixed-view anime tasks with white foreground on black."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "results/objectlike_anime_fixed_10cm_256_20260708"
OUTPUT = ROOT / "results/table2_anime_fixed3_inputs_20260722"
GROUPS = ("arcane_casters", "blade_heroes", "stage_dancers")


def main() -> None:
    rows = []
    for group in GROUPS:
        task_dir = OUTPUT / group
        task_dir.mkdir(parents=True, exist_ok=True)
        views = []
        for index in range(3):
            source = SOURCE / group / "inputs6" / f"input_view{index}.png"
            image = np.asarray(Image.open(source).convert("L"))
            # Source assets use black silhouettes on a white panel. The voxel
            # trainer expects white foreground on black.
            mask = image < 128
            target = (mask.astype(np.uint8) * 255)
            destination = task_dir / f"view_{index}.png"
            Image.fromarray(target, mode="L").save(destination)
            views.append(str(destination))
        rows.append({"task": group, "views": views})
    manifest = {
        "protocol": "fixed-3-view TFS, mr=1, 256^3",
        "foreground": "white-on-black",
        "tasks": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
