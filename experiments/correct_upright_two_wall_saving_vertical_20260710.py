#!/usr/bin/env python3
"""Re-export quality-first two-wall candidates with physical +Z upright."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/anime_two_wall_fixed_no_floor_wall_aligned_qualityfirst_saving_10cm_256_20260710"
OUT = ROOT / "results/anime_two_wall_fixed_no_floor_wall_aligned_qualityfirst_upright_10cm_256_20260710"
BASE = ROOT / "experiments/build_anime_n2_fixed_display_10cm_256_20260709.py"

spec = importlib.util.spec_from_file_location("n2", str(BASE))
n2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(n2)


def main() -> None:
    source_manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for record in source_manifest["cases"]:
        if not record.get("accepted"):
            continue
        name = record["case"]
        destination = OUT / name
        destination.mkdir(parents=True, exist_ok=True)
        # Existing field axes are [image-row(z), y, x]. Image row zero is the
        # visual top; reverse it before STL extraction so it becomes +Z.
        volume = np.load(record["voxels"]).astype(bool)
        upright = np.flip(volume, axis=0).copy()
        stl = destination / f"{name}_physically_upright_saving_10cm_256.stl"
        mesh = n2.export_stl(upright, stl)
        voxels = destination / f"{name}_physically_upright_saving_voxels256.npy"
        np.save(voxels, upright.astype(np.uint8))
        for folder in ("inputs", "projections"):
            shutil.copytree(SRC / name / folder, destination / folder, dirs_exist_ok=True)
        shutil.copy2(SRC / name / "input_vs_projection.png", destination / "input_vs_projection.png")
        updated = dict(record)
        updated.update({
            "setting": "two-wall fixed, no-floor, physically upright wall-aligned, quality-first material-saving",
            "physical_axis_mapping": "image top -> physical +Z; target left/right retain their calibrated wall axes",
            "stl": str(stl),
            "voxels": str(voxels),
            "mesh_parts": int(len(mesh.split(only_watertight=False))),
            "mesh_watertight": bool(mesh.is_watertight),
        })
        (destination / "CASE_MANIFEST.json").write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(updated)
    manifest = {
        "schema": "two_wall_qualityfirst_physical_upright_v1",
        "source": str(SRC),
        "n_cases": len(records),
        "cases": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "n_cases": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
