"""
体素到网格导出

功能:
1. Marching Cubes → 表面网格
2. Laplacian平滑
3. STL/OBJ导出
"""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def voxel_to_mesh(
    density: np.ndarray,
    threshold: float = 0.5,
    smooth_sigma: float = 1.0,
    smooth_iterations: int = 3,
    voxel_size: float = 1.0,
    fill_internal_holes: bool = False,
    keep_largest_component: bool = False,
) -> tuple:
    """体素场 → 三角网格

    Args:
        density: [N, N, N] 密度场
        threshold: 二值化阈值
        smooth_sigma: 高斯平滑sigma
        smooth_iterations: Laplacian平滑迭代次数
        voxel_size: 体素物理尺寸 (mm)
    Returns:
        (vertices, faces) — vertices [V, 3], faces [F, 3]
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for marching cubes")

    from scipy.ndimage import binary_fill_holes, gaussian_filter

    volume = density.astype(np.float64)
    if fill_internal_holes:
        volume = binary_fill_holes(volume > threshold).astype(np.float64)

    # 高斯平滑
    vol_smooth = gaussian_filter(volume, sigma=smooth_sigma)

    # Marching Cubes
    try:
        verts, faces, normals, values = marching_cubes(vol_smooth, level=threshold)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Marching cubes failed: {e}")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    # 缩放到物理尺寸
    verts = verts * voxel_size

    if HAS_TRIMESH and len(verts) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if keep_largest_component:
            parts = mesh.split(only_watertight=False)
            if len(parts) > 1:
                mesh = max(parts, key=lambda part: len(part.faces))
        if smooth_iterations > 0 and len(mesh.vertices) > 0:
            trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iterations)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        # Laplacian平滑
        if smooth_iterations > 0 and len(verts) > 0:
            verts = _laplacian_smooth(verts, faces, iterations=smooth_iterations)

    return verts, faces


def _laplacian_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    lamb: float = 0.5,
) -> np.ndarray:
    """Laplacian平滑"""
    from collections import defaultdict

    n_verts = len(verts)
    adj = defaultdict(set)
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(n_verts):
            if adj[i]:
                neighbors = list(adj[i])
                avg = verts[neighbors].mean(axis=0)
                new_verts[i] = verts[i] + lamb * (avg - verts[i])
        verts = new_verts

    return verts


def voxel_to_stl(
    density: np.ndarray,
    output_path: str,
    threshold: float = 0.5,
    voxel_size: float = 1.0,
    smooth_iterations: int = 3,
    fill_internal_holes: bool = False,
    keep_largest_component: bool = False,
) -> str:
    """导出STL文件"""
    verts, faces = voxel_to_mesh(
        density, threshold=threshold,
        smooth_iterations=smooth_iterations,
        voxel_size=voxel_size,
        fill_internal_holes=fill_internal_holes,
        keep_largest_component=keep_largest_component,
    )

    if len(verts) == 0:
        print("WARNING: Empty mesh, no STL written")
        return ""

    if HAS_TRIMESH:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(output_path)
    else:
        _write_stl_binary(verts, faces, output_path)

    print(f"Exported STL: {len(verts)} vertices, {len(faces)} faces → {output_path}")
    return output_path


def voxel_to_obj(
    density: np.ndarray,
    output_path: str,
    threshold: float = 0.5,
    voxel_size: float = 1.0,
    smooth_iterations: int = 3,
    fill_internal_holes: bool = False,
    keep_largest_component: bool = False,
) -> str:
    """导出OBJ文件"""
    verts, faces = voxel_to_mesh(
        density, threshold=threshold,
        smooth_iterations=smooth_iterations,
        voxel_size=voxel_size,
        fill_internal_holes=fill_internal_holes,
        keep_largest_component=keep_largest_component,
    )

    if len(verts) == 0:
        print("WARNING: Empty mesh, no OBJ written")
        return ""

    with open(output_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Exported OBJ: {len(verts)} vertices, {len(faces)} faces → {output_path}")
    return output_path


def _write_stl_binary(verts: np.ndarray, faces: np.ndarray, path: str):
    """简单的二进制STL写入"""
    import struct

    n_faces = len(faces)
    with open(path, 'wb') as f:
        # 80字节头
        f.write(b'\0' * 80)
        # 面片数
        f.write(struct.pack('<I', n_faces))

        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            # 法线 (简单叉乘)
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal /= norm_len

            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))  # attribute byte count
