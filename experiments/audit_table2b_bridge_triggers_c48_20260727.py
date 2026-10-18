#!/usr/bin/env python3
"""Screen frozen proposals for bridge-sensitive Table 2b tasks at C48."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT))

from shadowforge_pipeline.unified_run import S6, downsample_max_to
from voxel_ops import connect6_min_bridges


CASES = {
    "future_mech": ROOT / "results/soft_novel_loss_ablations_v3_20260715/future_mech/full/seed_17",
    "sword_fox": ROOT / "results/soft_novel_loss_ablations_v3_20260715/sword_fox/full/seed_17",
    "parrot_snail": ROOT / "results/soft_novel_loss_ablations_v3_20260715/parrot_snail/full/seed_17",
    "owl_turtle": ROOT / "results/soft_novel_loss_ablations_v3_20260715/owl_turtle/full/seed_17",
    "dark_knight": ROOT / "results/dark_knight_protocol_exact_rerun_20260726/soft_training",
    "aaai_maple": ROOT / "results/aaai_maple_printsafe_display_100mm_r1_20260722/soft_training",
}
OUT = ROOT / "results/table2b_bridge_trigger_c48_audit_20260727.json"


def component_stats(volume: np.ndarray) -> tuple[int, float]:
    labels, count = label(volume, structure=S6)
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / max(int(volume.sum()), 1))


rows = []
for name, path in CASES.items():
    probability = np.load(path / "occupancy_probability_core.npy", mmap_mode="r")
    envelope = np.load(path / "occupancy_soft_proposal.npy", mmap_mode="r")
    guidance = downsample_max_to(probability, 48)
    hull = downsample_max_to(envelope, 48) > 0
    for threshold in (0.15, 0.35, 0.55):
        seed = (guidance >= threshold) & hull
        before_count, before_largest = component_stats(seed)
        repaired, bridge_voxels = connect6_min_bridges(
            seed, hull, max_passes=8, bridge_radius=1
        )
        after_count, after_largest = component_stats(repaired)
        rows.append(
            {
                "task": name,
                "threshold": threshold,
                "seed_voxels": int(seed.sum()),
                "components_before": before_count,
                "largest_fraction_before": before_largest,
                "bridge_voxels": int(bridge_voxels),
                "components_after": after_count,
                "largest_fraction_after": after_largest,
                "bridge_triggered": bool(bridge_voxels > 0),
                "bridge_closed_components": bool(after_count < before_count),
            }
        )

summary = {
    "purpose": "C48 trigger screen only; use selected tasks for full delivery ablation",
    "rows": rows,
    "triggered_tasks": sorted({row["task"] for row in rows if row["bridge_triggered"]}),
    "component_sensitive_tasks": sorted(
        {row["task"] for row in rows if row["bridge_closed_components"]}
    ),
}
OUT.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
