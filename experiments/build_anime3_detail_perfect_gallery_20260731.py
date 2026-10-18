#!/usr/bin/env python3
"""Build three isolated, detail-first free-form anime P-CRAFT examples.

This wrapper reuses the validated July 7 free-form delivery path while
restricting material removal to preserve thin silhouette details. Historical
outputs and the mature two-view path are never modified.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "experiments/build_anime30_diverse_fixed_50_60_10cm_256_20260707.py"
OUT = ROOT / "results/pcraft_anime3_detail_perfect_20260731"

EXAMPLES = [
    ("detail_archer_miko_winged", "thick1", ["anime30_09", "anime30_28", "anime30_29"]),
    ("detail_fairy_mecha_scout", "close2", ["anime30_08", "anime30_14", "anime30_30"]),
    ("detail_mage_swordswoman_knight", "thick1", ["anime30_13", "anime30_21", "anime30_23"]),
]

WORST_VIEW_IOU_MIN = 0.985


def load_builder():
    spec = importlib.util.spec_from_file_location("pcraft_detail_builder", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load free-form builder: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.IOU_MIN = 0.983
    module.SAVE_MIN = 0.30
    module.SAVE_MAX = 0.40
    module.EROSION_MIN = 0.15
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
        strict_detail_pass = bool(
            record["accepted"]
            and min(selected["view_ious"]) >= WORST_VIEW_IOU_MIN
            and selected["mesh_watertight"]
            and selected["mesh_parts"] == 1
        )
        record["accepted_by_detail_gate"] = strict_detail_pass
        record["delivery_family"] = "freeform_projection_preserving_carving"
        record["cube_carving_branch"] = False
        records.append(record)
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, indent=2))
        print(
            f"{example}: detail_gate={strict_detail_pass} "
            f"candidate={selected['candidate']} mean={selected['mean_iou']:.6f} "
            f"worst={min(selected['view_ious']):.6f} saving={selected['saving']:.4f} "
            f"parts={selected['mesh_parts']} watertight={selected['mesh_watertight']}",
            flush=True,
        )

    summary = {
        "schema": "pcraft_anime3_detail_perfect_v1",
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
