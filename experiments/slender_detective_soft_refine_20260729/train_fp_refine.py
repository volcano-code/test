#!/usr/bin/env python3
"""Isolated soft-proposal refinement with a stronger exterior-leakage penalty."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


VOXEL_ROOT = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/"
    "XUXINYAO/shadow-project2/mesh/voxel"
)


def main() -> None:
    fp_scale = float(os.environ.get("PCRAFT_FP_SCALE", "2.0"))
    sys.path.insert(0, str(VOXEL_ROOT))

    import training_scheduler_voxel

    original_stage_weights = training_scheduler_voxel.stage_weights

    def refined_stage_weights(step: int, total: int) -> dict[str, float]:
        weights = original_stage_weights(step, total)
        weights["false_positive"] *= fp_scale
        return weights

    training_scheduler_voxel.stage_weights = refined_stage_weights
    runpy.run_path(str(VOXEL_ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
