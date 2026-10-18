#!/usr/bin/env python3
"""Create lower-density print meshes from an already validated exterior shell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymeshlab
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, nargs="+", default=[250000, 150000])
    return parser.parse_args()


def audit(path: Path) -> dict:
    mesh = trimesh.load_mesh(path, process=True)
    parts = list(mesh.split(only_watertight=False))
    return {
        "path": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(parts)),
        "extents_mm": [float(value) for value in mesh.extents],
        "volume_mm3": float(abs(mesh.volume)),
    }


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [{"variant": "source", **audit(source)}]
    for target_faces in args.target_faces:
        mesh_set = pymeshlab.MeshSet()
        mesh_set.load_new_mesh(str(source))
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            autoclean=True,
        )
        output = output_dir / f"{source.stem}_qem_{target_faces}f.stl"
        mesh_set.save_current_mesh(str(output), binary=True)
        records.append(
            {
                "variant": f"qem_{target_faces}",
                **audit(output),
            }
        )

    (output_dir / "MANIFEST.json").write_text(
        json.dumps({"source": str(source), "variants": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
