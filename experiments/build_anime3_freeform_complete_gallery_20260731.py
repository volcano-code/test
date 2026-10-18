#!/usr/bin/env python3
"""Build three isolated free-form anime three-view P-CRAFT examples.

This wrapper deliberately reuses the validated July 7 free-form delivery path.
It does not invoke the later cube-carving gallery branch and never writes into
the historical result directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "experiments/build_anime30_diverse_fixed_50_60_10cm_256_20260707.py"
OUT = ROOT / "results/pcraft_freeform_complete_anime3_20260731"

EXAMPLES = [
    ("sword_violin_miko", "thick1", ["anime30_03", "anime30_17", "anime30_28"]),
    ("adventurer_scholar_warrior", "thick1", ["anime30_11", "anime30_19", "anime30_26"]),
    ("future_miko_mage", "thick1", ["anime30_05", "anime30_28", "anime30_23"]),
]


def load_builder():
    spec = importlib.util.spec_from_file_location("pcraft_freeform_builder", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load free-form builder: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.IOU_MIN = 0.970
    module.SAVE_MIN = 0.30
    module.SAVE_MAX = 0.50
    module.EROSION_MIN = 0.15
    return module


def write_result_indexes(records) -> None:
    """Keep accepted free-form deliveries separate from rejected probes."""
    accepted = [record for record in records if record.get("accepted") is True]
    rejected = [record for record in records if record.get("accepted") is not True]
    (OUT / "ACCEPTED_MANIFEST.json").write_text(json.dumps(accepted, indent=2))
    (OUT / "REJECTED_PROBES.json").write_text(json.dumps(rejected, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in EXAMPLES],
        help="Run one pilot example instead of the complete gallery.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep other completed examples in the isolated output directory.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rebuild accepted/rejected indexes without rerunning geometry generation.",
    )
    args = parser.parse_args()

    builder = load_builder()
    if args.index_only:
        source = OUT / "MANIFEST.partial.json"
        if not source.exists():
            raise FileNotFoundError(source)
        write_result_indexes(json.loads(source.read_text()))
        print(f"INDEXED {OUT}", flush=True)
        return

    selected = [item for item in EXAMPLES if args.only is None or item[0] == args.only]

    if OUT.exists() and not args.keep_output:
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    partial = OUT / "MANIFEST.partial.json"
    if args.keep_output and partial.exists():
        records = json.loads(partial.read_text())
        replaced = {item[0] for item in selected}
        records = [record for record in records if record["example"] not in replaced]

    for example, mode, names in selected:
        record = builder.run_one(example, mode, names)
        record["delivery_family"] = "freeform_projection_preserving_carving"
        record["cube_carving_branch"] = False
        records.append(record)
        partial.write_text(json.dumps(records, indent=2))
        chosen = record["selected"]
        print(
            f"{example}: accepted={record['accepted']} candidate={chosen['candidate']} "
            f"IoU={chosen['mean_iou']:.6f} worst={chosen['min_iou']:.6f} "
            f"saving={chosen['saving']:.4f} parts={chosen['mesh_parts']} "
            f"watertight={chosen['mesh_watertight']}",
            flush=True,
        )

    summary = {
        "schema": "pcraft_freeform_complete_anime3_v1",
        "delivery_family": "freeform_projection_preserving_carving",
        "cube_carving_branch": False,
        "resolution": 256,
        "print_mm": 100.0,
        "gate": {
            "mean_iou_min": builder.IOU_MIN,
            "saving_range": [builder.SAVE_MIN, builder.SAVE_MAX],
            "single": True,
            "watertight": True,
            "erosion_survival_min": builder.EROSION_MIN,
        },
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    write_result_indexes(records)
    print(f"DONE {OUT}", flush=True)


if __name__ == "__main__":
    main()
