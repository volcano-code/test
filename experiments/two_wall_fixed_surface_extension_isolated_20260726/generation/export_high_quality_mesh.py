"""
Export a higher-quality mesh from a saved voxel volume.

Default behavior uses marching cubes on a smoothed occupancy field, which is
better aligned with binary voxel sculpture data than Poisson reconstruction.
Poisson is provided as an optional mode when Open3D is installed.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from voxel_ops import clean_binary_volume

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def _load_volume(path: Path) -> np.ndarray:
    vol = np.load(path)
    if vol.ndim == 4:
        vol = vol[0]
    return (vol > 0.5).astype(np.float32)


def _to_world_coords(verts: np.ndarray, volume_shape, extent_world: float) -> np.ndarray:
    verts = verts.astype(np.float32)
    denom = np.maximum(np.array(volume_shape, dtype=np.float32) - 1.0, 1.0)
    return verts / denom * extent_world - extent_world / 2.0


def _save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def export_marching_cubes(
    volume: np.ndarray,
    out_path: Path,
    extent_world: float = 1.7,
    smooth_sigma: float = 0.35,
    taubin_iterations: int = 2,
    upsample_factor: int = 2,
) -> None:
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required for marching cubes export")

    cleaned = clean_binary_volume(
        volume > 0.5,
        fill_holes=True,
        closing_iters=1,
        opening_iters=0,
        keep_largest=True,
    ).astype(np.float32)

    if int(upsample_factor) > 1:
        cleaned = zoom(
            cleaned,
            zoom=(int(upsample_factor), int(upsample_factor), int(upsample_factor)),
            order=0,
        ).astype(np.float32)

    padded = np.pad(cleaned, 1, mode="constant")
    smooth = gaussian_filter(padded, sigma=float(smooth_sigma))
    verts, faces, _, _ = marching_cubes(smooth, level=0.5)
    verts = verts - 1.0
    verts = _to_world_coords(verts, cleaned.shape, extent_world)
    faces = faces.astype(np.int64)

    if HAS_TRIMESH:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(mesh, iterations=int(taubin_iterations))
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.export(out_path)
    else:
        _save_obj(out_path, verts, faces)


def export_poisson(
    volume: np.ndarray,
    out_path: Path,
    extent_world: float = 1.7,
    depth: int = 9,
    smooth_sigma: float = 0.9,
) -> None:
    if not HAS_OPEN3D:
        raise RuntimeError("open3d is required for Poisson reconstruction")
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required to seed Poisson reconstruction")

    cleaned = clean_binary_volume(
        volume > 0.5,
        fill_holes=True,
        closing_iters=1,
        opening_iters=0,
        keep_largest=True,
    ).astype(np.float32)

    padded = np.pad(cleaned, 1, mode="constant")
    smooth = gaussian_filter(padded, sigma=float(smooth_sigma))
    verts, faces, _, _ = marching_cubes(smooth, level=0.5)
    verts = verts - 1.0
    verts = _to_world_coords(verts, cleaned.shape, extent_world)

    seed_mesh = o3d.geometry.TriangleMesh()
    seed_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    seed_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    seed_mesh.compute_vertex_normals()
    pcd = seed_mesh.sample_points_poisson_disk(number_of_points=max(len(verts), 20000))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(depth),
    )
    densities = np.asarray(densities)
    if densities.size > 0:
        keep = densities >= np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(str(out_path), mesh)


def export_reconstructed_mesh(
    volume: np.ndarray,
    out_path,
    method: str = "mcubes",
    extent_world: float = 1.7,
    smooth_sigma: float = 0.35,
    taubin_iterations: int = 2,
    upsample_factor: int = 2,
    poisson_depth: int = 9,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if method == "poisson":
        export_poisson(
            volume,
            out_path,
            extent_world=extent_world,
            depth=poisson_depth,
            smooth_sigma=smooth_sigma,
        )
    else:
        export_marching_cubes(
            volume,
            out_path,
            extent_world=extent_world,
            smooth_sigma=smooth_sigma,
            taubin_iterations=taubin_iterations,
            upsample_factor=upsample_factor,
        )


def main():
    parser = argparse.ArgumentParser(description="Export a higher-quality mesh from voxel occupancy")
    parser.add_argument("--input", required=True, help="Path to saved voxel npy")
    parser.add_argument("--output", required=True, help="Output OBJ/PLY mesh path")
    parser.add_argument("--method", choices=["mcubes", "poisson"], default="mcubes")
    parser.add_argument("--extent_world", type=float, default=1.7)
    parser.add_argument("--smooth_sigma", type=float, default=0.35)
    parser.add_argument("--taubin_iterations", type=int, default=2)
    parser.add_argument("--upsample_factor", type=int, default=2)
    parser.add_argument("--poisson_depth", type=int, default=9)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    volume = _load_volume(input_path)
    export_reconstructed_mesh(
        volume,
        out_path,
        method=args.method,
        extent_world=float(args.extent_world),
        smooth_sigma=float(args.smooth_sigma),
        taubin_iterations=int(args.taubin_iterations),
        upsample_factor=int(args.upsample_factor),
        poisson_depth=int(args.poisson_depth),
    )
    print(f"Saved reconstructed mesh ({args.method}) -> {out_path}")


if __name__ == "__main__":
    main()
