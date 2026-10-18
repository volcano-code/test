"""
Closed-loop safe OBJ export:
1. Export a conservative / low-smoothing OBJ from a verified high-res voxel volume.
2. Re-voxelize the OBJ back into the original high-res grid.
3. Re-run FEM verification on the re-voxelized volume.
4. If needed, increase conservative pre-export thickening and retry until SF >= target.

This script does not overwrite the current pipeline outputs. It writes into a new
output directory so previous results remain available for rollback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, zoom

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural.analysis.fem_solver import quick_sf_check
from structural.export.remap_fem_repair_to_highres import (
    _load_binary_volume,
    evaluate_silhouette_changes,
    save_difference_panel,
    save_slice_panel,
    summarize_volume_changes,
)
from structural.export.voxel_to_mesh import voxel_to_mesh


def _write_obj(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _mesh_part_count(obj_path: Path) -> int:
    import trimesh

    mesh = trimesh.load_mesh(obj_path, process=False)
    return len(mesh.split(only_watertight=False))


def export_obj_candidate(
    volume: np.ndarray,
    obj_path: Path,
    threshold: float,
    smooth_sigma: float,
    smooth_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vertices, faces = voxel_to_mesh(
        density=volume.astype(np.float32),
        threshold=threshold,
        smooth_sigma=smooth_sigma,
        smooth_iterations=smooth_iterations,
        voxel_size=1.0,
        fill_internal_holes=True,
        keep_largest_component=True,
    )
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("OBJ export failed: empty mesh")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(vertices, faces, obj_path)
    return vertices, faces


def revoxelize_obj_to_grid(obj_path: Path, target_shape: Tuple[int, int, int]) -> np.ndarray:
    import trimesh

    mesh = trimesh.load_mesh(obj_path, process=False)
    vox = mesh.voxelized(pitch=1.0).fill()
    points = np.asarray(vox.points)
    coords = np.rint(points).astype(int)
    coords = np.clip(coords, 0, np.array(target_shape) - 1)
    volume = np.zeros(target_shape, dtype=np.float32)
    if len(coords) > 0:
        volume[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    volume = binary_fill_holes(volume > 0.5).astype(np.float32)
    return volume


def verify_high_volume(high_volume: np.ndarray) -> Dict[str, object]:
    low_density = zoom(
        high_volume.astype(np.float32),
        (64 / high_volume.shape[0],) * 3,
        order=1,
    )
    low_density = np.clip(low_density, 0.0, 1.0)
    fem = quick_sf_check(low_density.astype(np.float32), verbose=False)
    solid = low_density > 0.3
    vm_solid = fem["von_mises"][solid]
    disp_max = float(np.abs(fem["displacement"]).max())
    return {
        "downsampled_density": low_density,
        "fem": fem,
        "sf_min": float(fem["sf_min"]),
        "vm_max": float(vm_solid.max()) if vm_solid.size else 0.0,
        "disp_max": disp_max,
    }


def save_verification_outputs(out_dir: Path, verify: Dict[str, object]) -> None:
    fem = verify["fem"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "grid_size": int(verify["downsampled_density"].shape[0]),
        "solid_ratio": float((verify["downsampled_density"] > 0.5).mean()),
        "fem": {
            "sf_min": float(verify["sf_min"]),
            "vm_max": float(verify["vm_max"]),
            "vm_mean": float(fem["von_mises"][verify["downsampled_density"] > 0.3].mean())
            if np.any(verify["downsampled_density"] > 0.3)
            else 0.0,
            "disp_max": float(verify["disp_max"]),
            "n_elements": int(fem["n_elements"]),
            "n_dofs": int(fem["n_dofs"]),
            "solve_time_s": float(fem["solve_time_s"]),
            "total_time_s": float(fem["total_time_s"]),
        },
    }
    (out_dir / "fem_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "fem_von_mises.npy", fem["von_mises"])
    np.save(out_dir / "fem_displacement.npy", fem["displacement"])
    np.save(out_dir / "downsampled_density.npy", verify["downsampled_density"].astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a closed-loop safe OBJ with FEM re-verification")
    parser.add_argument("--source_volume", required=True, help="Verified high-res source voxel volume")
    parser.add_argument("--reference_volume", required=True, help="Original high-res volume for silhouette comparison")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_sf", type=float, default=1.5)
    parser.add_argument("--thresholds", default="0.50,0.45,0.40")
    parser.add_argument("--smooth_sigmas", default="0.00,0.25,0.50,1.00")
    parser.add_argument("--smooth_iterations", default="0,1,2")
    parser.add_argument("--dilation_iters", default="0,1,2")
    args = parser.parse_args()

    source_high = _load_binary_volume(Path(args.source_volume))
    reference_high = _load_binary_volume(Path(args.reference_volume))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_dir = out_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    smooth_sigmas = [float(x) for x in args.smooth_sigmas.split(",") if x.strip()]
    smooth_iterations = [int(x) for x in args.smooth_iterations.split(",") if x.strip()]
    dilation_iters = [int(x) for x in args.dilation_iters.split(",") if x.strip()]

    all_results: List[Dict[str, object]] = []
    chosen: Dict[str, object] | None = None

    for dil in dilation_iters:
        export_volume = source_high if dil == 0 else binary_dilation(source_high, iterations=dil)
        safe_this_level: List[Dict[str, object]] = []

        for sigma in smooth_sigmas:
            for smooth_it in smooth_iterations:
                for threshold in thresholds:
                    cand_name = f"dil{dil}_sig{sigma:.2f}_sm{smooth_it}_thr{threshold:.2f}".replace(".", "p")
                    cand_dir = search_dir / cand_name
                    obj_path = cand_dir / "candidate.obj"
                    cand_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        export_obj_candidate(
                            volume=export_volume,
                            obj_path=obj_path,
                            threshold=threshold,
                            smooth_sigma=sigma,
                            smooth_iterations=smooth_it,
                        )
                        revoxelized = revoxelize_obj_to_grid(obj_path, source_high.shape)
                        verify = verify_high_volume(revoxelized)
                        silhouette = evaluate_silhouette_changes(reference_high, revoxelized > 0.5)
                        add_mask = (revoxelized > 0.5) & (~reference_high)
                        volume = summarize_volume_changes(reference_high, revoxelized > 0.5, add_mask)
                        parts = _mesh_part_count(obj_path)
                        changed_pixels = int(sum(v["changed_pixels"] for v in silhouette["per_view"].values()))
                        candidate = {
                            "candidate": cand_name,
                            "dilation_iters": dil,
                            "smooth_sigma": sigma,
                            "smooth_iterations": smooth_it,
                            "threshold": threshold,
                            "sf_min": float(verify["sf_min"]),
                            "vm_max": float(verify["vm_max"]),
                            "disp_max": float(verify["disp_max"]),
                            "changed_pixels": changed_pixels,
                            "volume_ratio": float(volume["volume_ratio"]),
                            "mesh_parts": int(parts),
                            "safe": bool(verify["sf_min"] >= args.target_sf and parts == 1),
                            "obj_path": str(obj_path),
                            "candidate_dir": str(cand_dir),
                        }
                        (cand_dir / "candidate_summary.json").write_text(
                            json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        all_results.append(candidate)
                        if candidate["safe"]:
                            safe_this_level.append({**candidate, "_verify": verify, "_revoxelized": revoxelized})
                    except Exception as e:
                        candidate = {
                            "candidate": cand_name,
                            "dilation_iters": dil,
                            "smooth_sigma": sigma,
                            "smooth_iterations": smooth_it,
                            "threshold": threshold,
                            "error": str(e),
                            "safe": False,
                        }
                        (cand_dir / "candidate_summary.json").write_text(
                            json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        all_results.append(candidate)

        if safe_this_level:
            chosen = min(
                safe_this_level,
                key=lambda c: (c["changed_pixels"], c["volume_ratio"], c["smooth_iterations"], c["smooth_sigma"]),
            )
            break

    if chosen is None:
        (out_dir / "search_summary.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError("No closed-loop safe OBJ candidate reached the target SF.")

    final_obj = out_dir / "closed_loop_safe.obj"
    final_revox = out_dir / "closed_loop_safe_revoxelized.npy"
    Path(chosen["obj_path"]).replace(final_obj)
    np.save(final_revox, chosen["_revoxelized"].astype(np.float32)[None, ...])

    add_mask = (chosen["_revoxelized"] > 0.5) & (~reference_high)
    remove_mask = reference_high & (~(chosen["_revoxelized"] > 0.5))
    save_difference_panel(
        original_high=reference_high,
        repaired_high=chosen["_revoxelized"] > 0.5,
        add_mask=add_mask,
        remove_mask=remove_mask,
        out_path=out_dir / "closed_loop_difference_projection_panel.png",
    )
    save_slice_panel(
        original_high=reference_high,
        repaired_high=chosen["_revoxelized"] > 0.5,
        add_mask=add_mask,
        remove_mask=remove_mask,
        out_path=out_dir / "closed_loop_difference_slice_panel.png",
    )
    save_verification_outputs(out_dir / "fem_verify", chosen["_verify"])

    summary = {
        "backup_required": False,
        "target_sf": float(args.target_sf),
        "chosen_candidate": {
            k: v
            for k, v in chosen.items()
            if not k.startswith("_")
        },
        "search_count": len(all_results),
        "search_results": all_results,
        "artifacts": {
            "closed_loop_safe_obj": str(final_obj),
            "closed_loop_safe_revoxelized": str(final_revox),
            "fem_verify_dir": str(out_dir / "fem_verify"),
            "difference_projection_panel": str(out_dir / "closed_loop_difference_projection_panel.png"),
            "difference_slice_panel": str(out_dir / "closed_loop_difference_slice_panel.png"),
        },
    }
    (out_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["chosen_candidate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
