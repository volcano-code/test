"""Resolve a voxel input path into (density, force, fixed_mask).

Accepts .npy (density grid), .npz (expects `density` or `voxels` key), or a
surface mesh (.obj/.stl) which is voxelized via trimesh.

Non-native resolutions are automatically resampled to ``target_resolution``
(default 64) so the PINO surrogate always receives inputs at the resolution
it was trained on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from structural.analysis.load_generator import LoadGenerator

# PINO training resolution — all inputs are resampled to this before inference.
_TARGET_RESOLUTION: int = 64


def _resize_density(d: np.ndarray, target: int) -> np.ndarray:
    """Resample a 3-D density grid to (target, target, target)."""
    if d.shape == (target, target, target):
        return d
    from scipy.ndimage import zoom
    factors = tuple(target / s for s in d.shape)
    out = zoom(d, factors, order=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _load_density(path: Path, grid_size: int = _TARGET_RESOLUTION) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        d = np.load(path)
    elif suf == ".npz":
        blob = np.load(path)
        key = next((k for k in ["density", "voxels", "voxel", "grid"] if k in blob), None)
        if key is None:
            raise ValueError(f"{path}: no density/voxels/voxel/grid key in npz")
        d = blob[key]
    elif suf in (".obj", ".stl", ".ply"):
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        vox = mesh.voxelized(pitch=max(mesh.extents) / grid_size)
        vox = vox.matrix.astype(np.float32)
        # Pad/crop to cubic grid_size
        d = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        s = [min(vox.shape[i], grid_size) for i in range(3)]
        d[:s[0], :s[1], :s[2]] = vox[:s[0], :s[1], :s[2]]
    else:
        raise ValueError(f"Unsupported input format: {suf}")

    d = np.asarray(d, dtype=np.float32)
    if d.ndim == 4 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 3:
        raise ValueError(f"{path}: density must be 3D, got shape {d.shape}")

    # Auto-resize to training resolution
    if d.shape[0] != grid_size or d.shape[1] != grid_size or d.shape[2] != grid_size:
        src_shape = d.shape
        d = _resize_density(d, grid_size)
        print(f"[load_resolver] resampled {src_shape} → {d.shape}")

    return d


def resolve(
    path: str | Path,
    load_case: str = "standing",
    force: np.ndarray | None = None,
    fixed_mask: np.ndarray | None = None,
    material_density: float = 1240.0,
    gravity: float = -9.81,
    cell_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (density [N,N,N], force [3,N,N,N], fixed_mask [1,N,N,N])."""
    path = Path(path)
    density = _load_density(path)

    if load_case == "custom":
        if force is None or fixed_mask is None:
            raise ValueError("load_case='custom' requires force and fixed_mask")
        return density, force.astype(np.float32), fixed_mask.astype(np.float32)

    lg = LoadGenerator(
        gravity=gravity,
        material_density=material_density,
        cell_size=cell_size,
    )
    if load_case == "standing":
        blob = lg.standing_load(density)
    elif load_case == "grip":
        blob = lg.gripping_load(density)
    else:
        raise ValueError(f"Unknown load_case: {load_case}")
    return density, blob["force"], blob["fixed_mask"]
