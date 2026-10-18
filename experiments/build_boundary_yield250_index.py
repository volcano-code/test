#!/usr/bin/env python3
"""Build a stronger boundary training index for the raw v29 PINO protocol.

Local aug_boundary samples were generated with a 50 MPa yield convention. The
old v29/Hard450 protocol uses 250 MPa. Because linear elasticity scales stress
linearly with load, multiplying force and vm_max by 5 preserves the original
SF labels under the 250 MPa convention.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TOTAL = Path("/data/xvxinyao/xvxinyao/total_project")
SRC = TOTAL / "results/shadowart_merged_detect_repair_v1/aug_boundary"
OLD_INDEX = TOTAL / "results/shadowart_fem_hardset_v1/index.json"
OLD_ROOT = OLD_INDEX.parent
OUT_ROOT = Path("/home/lin/xvxinyao/ShadowForge/results/boundary_fem_yield250_20260624")
OUT_SAMPLES = OUT_ROOT / "samples"
OUT_INDEX = OUT_ROOT / "index.json"
SCALE = 5.0


def convert_sample(src: Path, dst: Path) -> dict:
    data = np.load(src, allow_pickle=False)
    force = data["force"].astype(np.float32) * SCALE
    vm_max = np.float32(float(data["vm_max"]) * SCALE)
    sf = np.float32(float(data["sf"]))
    log_vm = np.float32(np.log10(max(float(vm_max), 1e-12)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        density=data["density"].astype(np.float32),
        force=force,
        fixed_mask=data["fixed_mask"].astype(np.float32),
        vm_max=vm_max,
        sf=sf,
        log_vm_max=log_vm,
        load_multiplier=np.float32(float(data["load_multiplier"]) * SCALE),
    )
    m = re.search(r"aug_(\d+)_sf([0-9.]+)\.npz$", src.name)
    group = int(m.group(1)) if m else -1
    return {
        "path": str(dst),
        "case_id": 100000 + group,
        "case_name": f"boundary_aug_{group:04d}",
        "split": "train",
        "sf": float(sf),
        "vm_max": float(vm_max),
        "log_vm_max": float(log_vm),
        "load_multiplier": float(data["load_multiplier"]) * SCALE,
        "source": "shadowart_merged_detect_repair_v1/aug_boundary_yield250",
    }


def main() -> None:
    old = json.loads(OLD_INDEX.read_text())
    def absolutize(records):
        out = []
        for rec in records:
            item = dict(rec)
            item["path"] = str(OLD_ROOT / item["path"])
            out.append(item)
        return out
    converted = []
    for src in sorted(SRC.glob("*.npz")):
        dst = OUT_SAMPLES / src.name
        converted.append(convert_sample(src, dst))

    index = {
        "config": {
            "schema_version": "boundary_yield250_index_v1",
            "old_index": str(OLD_INDEX),
            "source": str(SRC),
            "stress_scale": SCALE,
            "note": "force and vm_max scaled by 5 so original 50MPa SF labels match 250MPa protocol",
        },
        "cases": old.get("cases", []),
        "train": absolutize(old["train"]) + converted,
        "val": absolutize(old["val"]),
        "test": absolutize(old["test"]),
        "boundary_extra_train": converted,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text(json.dumps(index, indent=2))
    sfs = np.asarray([r["sf"] for r in converted])
    summary = {
        "index": str(OUT_INDEX),
        "n_old_train": len(old["train"]),
        "n_boundary_extra": len(converted),
        "n_train_total": len(index["train"]),
        "n_val": len(index["val"]),
        "n_test": len(index["test"]),
        "sf_min": float(sfs.min()),
        "sf_max": float(sfs.max()),
        "sf_counts": {str(v): int(np.sum(np.isclose(sfs, v))) for v in sorted(set(sfs.tolist()))},
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
