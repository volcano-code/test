#!/usr/bin/env python3
"""Build three detail-first free-form anime P-CRAFT deliveries.

The triples are selected by an input-only fixed-axis compatibility screen.
This isolated wrapper leaves the validated July 7 builder and all historical
outputs untouched.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "experiments/build_anime30_diverse_fixed_50_60_10cm_256_20260707.py"
OUT = ROOT / "results/pcraft_anime3_detail_perfect_v2_20260731"

# Ordered X-wall, Y-wall, floor/Z assignments from the input-only screen.
EXAMPLES = [
    ("detail_fox_warrior_mage", "close2", ["anime30_13", "anime30_29", "anime30_07"]),
    ("detail_swordswoman_warrior_mage", "close2", ["anime30_28", "anime30_29", "anime30_07"]),
    ("detail_fox_swordswoman_mage", "close2", ["anime30_13", "anime30_28", "anime30_07"]),
]

WORST_VIEW_IOU_MIN = 0.995


def load_builder():
    spec = importlib.util.spec_from_file_location("pcraft_detail_builder_v2", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load free-form builder: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.IOU_MIN = 0.995
    module.SAVE_MIN = 0.20
    module.SAVE_MAX = 0.40
    module.EROSION_MIN = 0.12
    return module


def main() -> None:
    builder = load_builder()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for example, mode, names in EXAMPLES:
        record = builder.run_one(example, mode, names)
        selected = record["selected"]
        strict_pass = bool(
            record["accepted"]
            and min(selected["view_ious"]) >= WORST_VIEW_IOU_MIN
            and selected["mesh_watertight"]
            and selected["mesh_parts"] == 1
        )
        record["accepted_by_detail_gate"] = strict_pass
        record["delivery_family"] = "freeform_projection_preserving_carving"
        record["cube_carving_branch"] = False
        records.append(record)
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, indent=2))
        print(
            f"{example}: detail_gate={strict_pass} "
            f"candidate={selected['candidate']} mean={selected['mean_iou']:.6f} "
            f"worst={min(selected['view_ious']):.6f} saving={selected['saving']:.4f} "
            f"parts={selected['mesh_parts']} watertight={selected['mesh_watertight']}",
            flush=True,
        )

    summary = {
        "schema": "pcraft_anime3_detail_perfect_v2",
        "selection_basis": "input_only_ordered_fixed_axis_compatibility",
        "delivery_family": "freeform_projection_preserving_carving",
        "cube_carving_branch": False,
        "resolution": 256,
        "print_mm": 100.0,
        "gate": {
            "mean_iou_min": builder.IOU_MIN,
            "worst_view_iou_min": WORST_VIEW_IOU_MIN,
            "saving_range": [builder.SAVE_MIN, builder.SAVE_MAX],
            "single": True,
            "watertight": True,
            "erosion_survival_min": builder.EROSION_MIN,
        },
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    accepted = [record for record in records if record["accepted_by_detail_gate"]]
    rejected = [record for record in records if not record["accepted_by_detail_gate"]]
    (OUT / "ACCEPTED_MANIFEST.json").write_text(json.dumps(accepted, indent=2))
    (OUT / "REJECTED_PROBES.json").write_text(json.dumps(rejected, indent=2))
    print(f"DONE accepted={len(accepted)}/{len(records)} output={OUT}", flush=True)


if __name__ == "__main__":
    main()
