#!/usr/bin/env python3
"""Generate normalized FEM von-Mises fields for unique candidate geometries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from structural.analysis.fem_solver import voxel_fem_solve
from structural.analysis.load_generator import LoadGenerator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/candidate_vm_fields_grouped_20260703.npz"


def main() -> None:
    generator = LoadGenerator()
    densities, fields, branches, groups, geometries = [], [], [], [], []
    vm_base, base_multipliers, valid_fields = [], [], []
    max_relative_error = 0.0
    mismatch_count = 0
    started = time.time()
    for branch_id, source in enumerate((ROOT / "data/aug.npz", ROOT / "data/aug_fixed.npz")):
        payload = np.load(source)
        if len(payload["dens"]) != 640:
            raise ValueError(f"Expected 640 rows in {source}")
        for base in range(0, 640, 4):
            density = payload["dens"][base].astype(np.float32)
            if not all(np.array_equal(density, payload["dens"][base + offset]) for offset in range(1, 4)):
                raise ValueError(f"Load replicas do not share geometry at {source}:{base}")
            load = generator.standing_load(density)
            multiplier = float(payload["mult"][base])
            result = voxel_fem_solve(
                density,
                load["force"] * multiplier,
                load["fixed_mask"],
                verbose=False,
            )
            vm = np.asarray(result["von_mises"], dtype=np.float32)
            stored_max = float(payload["vm"][base])
            computed_max = float(np.nanmax(vm[density > 0.5]))
            scale = max(abs(computed_max), abs(stored_max), 1e-30)
            relative_error = abs(computed_max - stored_max) / scale
            max_relative_error = max(max_relative_error, relative_error)
            mismatch_count += int(relative_error > 1e-6)
            valid = bool(np.isfinite(computed_max) and computed_max > 1e-6)
            densities.append(density.astype(np.uint8))
            fields.append((vm / max(computed_max, 1e-12)).astype(np.float16))
            branches.append(branch_id)
            groups.append(base // 16)
            geometries.append((base % 16) // 4)
            vm_base.append(computed_max)
            base_multipliers.append(multiplier)
            valid_fields.append(valid)
            if len(densities) % 40 == 0:
                print(json.dumps({"completed": len(densities), "elapsed_s": time.time() - started}), flush=True)
    np.savez_compressed(
        OUTPUT,
        density=np.asarray(densities, dtype=np.uint8),
        vm_normalized=np.asarray(fields, dtype=np.float16),
        branch=np.asarray(branches, dtype=np.uint8),
        group=np.asarray(groups, dtype=np.uint8),
        geometry=np.asarray(geometries, dtype=np.uint8),
        vm_base=np.asarray(vm_base, dtype=np.float64),
        base_multiplier=np.asarray(base_multipliers, dtype=np.float32),
        valid=np.asarray(valid_fields, dtype=bool),
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "n_unique_geometries": len(densities),
        "max_stored_vm_relative_error": max_relative_error,
        "stored_vm_mismatch_count": mismatch_count,
        "valid_field_count": int(sum(valid_fields)),
        "elapsed_s": time.time() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
