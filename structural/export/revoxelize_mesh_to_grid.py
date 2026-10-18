from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes


def revoxelize_mesh_to_grid(mesh_path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=False)
    mesh = mesh.copy()

    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    extents = np.maximum(maxs - mins, 1e-8)

    target = np.array(target_shape, dtype=np.float64)
    # Isotropic fit so geometry stays undistorted.
    scale = float(np.min((target - 3.0) / extents))
    mesh.vertices = (mesh.vertices - mins[None, :]) * scale + 1.0

    vox = mesh.voxelized(pitch=1.0).fill()
    points = np.asarray(vox.points)
    coords = np.rint(points).astype(int)
    coords = np.clip(coords, 0, np.array(target_shape) - 1)
    volume = np.zeros(target_shape, dtype=np.float32)
    if len(coords) > 0:
        volume[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    volume = binary_fill_holes(volume > 0.5).astype(np.float32)
    return volume


def main() -> None:
    parser = argparse.ArgumentParser(description="Revoxelize a mesh into a target voxel grid")
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--reference_npy", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    ref = np.load(args.reference_npy)
    if ref.ndim == 4:
        ref = ref[0]
    if ref.ndim == 5:
        ref = ref[0, 0]

    out = revoxelize_mesh_to_grid(Path(args.mesh_path), tuple(ref.shape))
    output_path = Path(args.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out.astype(np.float32))

    ref_bin = ref > 0.5
    out_bin = out > 0.5
    inter = int((ref_bin & out_bin).sum())
    union = int((ref_bin | out_bin).sum())
    iou = float(inter / union) if union else 1.0
    dice = float((2 * inter) / max(int(ref_bin.sum() + out_bin.sum()), 1))
    summary = {
        "mesh_path": str(Path(args.mesh_path)),
        "reference_npy": str(Path(args.reference_npy)),
        "output_npy": str(output_path),
        "reference_solid_ratio": float(ref_bin.mean()),
        "revoxelized_solid_ratio": float(out_bin.mean()),
        "iou": iou,
        "dice": dice,
        "added_voxels": int((out_bin & ~ref_bin).sum()),
        "removed_voxels": int((ref_bin & ~out_bin).sum()),
    }
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
