"""
Experimental direct mesh-FEM solver for OBJ/surface meshes.

Pipeline:
surface mesh -> TetGen tetrahedralization -> linear tetrahedral FEM ->
von Mises stress / displacement / safety factor.

This module is intentionally isolated from the existing voxel-FEM pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class TetMesh:
    nodes: np.ndarray  # [N, 3]
    elems: np.ndarray  # [M, 4]


def isotropic_material_matrix(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6), dtype=np.float64)
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def tetra_element_B_matrix(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    coords: [4, 3]
    returns B [6, 12], volume
    """
    X = np.ones((4, 4), dtype=np.float64)
    X[:, 1:] = coords
    detX = np.linalg.det(X)
    volume = abs(detX) / 6.0
    if volume < 1e-14:
        raise ValueError("Degenerate tetrahedron")

    invX = np.linalg.inv(X)
    grads = invX[1:, :]  # [3, 4], column i = grad N_i

    B = np.zeros((6, 12), dtype=np.float64)
    for i in range(4):
        bi, ci, di = grads[:, i]
        col = 3 * i
        B[0, col] = bi
        B[1, col + 1] = ci
        B[2, col + 2] = di
        B[3, col + 1] = di
        B[3, col + 2] = ci
        B[4, col] = di
        B[4, col + 2] = bi
        B[5, col] = ci
        B[5, col + 1] = bi
    return B, volume


def tetra_element_stiffness(coords: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    B, volume = tetra_element_B_matrix(coords)
    Ke = B.T @ D @ B * volume
    return Ke, volume, B


def tetrahedralize_surface_mesh(
    mesh_path: str | Path,
    target_tets: int = 30000,
    minratio: float = 1.8,
    mindihedral: float = 5.0,
    geometry_scale: float = 1.0,
    quality: bool = True,
    nobisect: bool = False,
    quiet: bool = True,
) -> TetMesh:
    import pyvista as pv
    import tetgen

    surface = pv.read(str(mesh_path)).triangulate().clean()
    if geometry_scale != 1.0:
        surface.points *= float(geometry_scale)
    tet = tetgen.TetGen(surface)

    bounds = np.asarray(surface.bounds, dtype=np.float64)
    bbox_volume = max((bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4]), 1e-8)
    maxvolume = bbox_volume / max(target_tets, 1)

    nodes, elems = tet.tetrahedralize(
        order=1,
        quality=bool(quality),
        minratio=minratio,
        mindihedral=mindihedral,
        maxvolume=maxvolume,
        nobisect=bool(nobisect),
        quiet=int(quiet),
    )
    elems = np.asarray(elems, dtype=np.int64)
    if elems.ndim == 2 and elems.shape[1] > 4:
        elems = elems[:, :4]
    return TetMesh(nodes=np.asarray(nodes, dtype=np.float64), elems=elems)


def generate_standing_load_for_tet_mesh(
    tet_mesh: TetMesh,
    material_density: float = 1.24e3,
    gravity: float = -9.81,
    support_thickness: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      force vector [3*N]
      fixed_dof_mask [3*N] bool
    """
    nodes = tet_mesh.nodes
    elems = tet_mesh.elems
    n_nodes = len(nodes)
    force = np.zeros(3 * n_nodes, dtype=np.float64)

    for tet in elems:
        coords = nodes[tet]
        try:
            _, volume, _ = tetra_element_stiffness(coords, np.eye(6))
        except Exception:
            continue
        f_node = material_density * gravity * volume / 4.0
        for nid in tet:
            force[3 * nid + 1] += f_node

    y = nodes[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    height = max(y_max - y_min, 1e-8)
    base_tol = float(support_thickness) if support_thickness is not None else max(1e-3, 0.02 * height)
    fixed_nodes = y <= (y_min + base_tol)
    fixed_mask = np.repeat(fixed_nodes, 3)
    return force, fixed_mask


def solve_tet_mesh_fem(
    tet_mesh: TetMesh,
    E: float = 3.5e9,
    nu: float = 0.36,
    yield_stress: float = 50.0e6,
    material_density: float = 1.24e3,
    gravity: float = -9.81,
    support_thickness: float | None = None,
) -> Dict[str, object]:
    nodes = tet_mesh.nodes
    elems = tet_mesh.elems
    n_nodes = len(nodes)
    total_dofs = 3 * n_nodes
    D = isotropic_material_matrix(E, nu)

    rows = []
    cols = []
    vals = []
    elem_B = np.zeros((len(elems), 6, 12), dtype=np.float64)
    elem_vol = np.zeros(len(elems), dtype=np.float64)
    valid_elem = np.zeros(len(elems), dtype=bool)

    for ei, tet in enumerate(elems):
        coords = nodes[tet]
        try:
            Ke, volume, B = tetra_element_stiffness(coords, D)
        except ValueError:
            continue
        elem_B[ei] = B
        elem_vol[ei] = volume
        valid_elem[ei] = True
        dofs = np.repeat(3 * tet, 3) + np.tile(np.arange(3), 4)
        rr = np.repeat(dofs, 12)
        cc = np.tile(dofs, 12)
        rows.append(rr)
        cols.append(cc)
        vals.append(Ke.reshape(-1))

    K = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total_dofs, total_dofs),
    ).tocsr()

    F, fixed_mask = generate_standing_load_for_tet_mesh(
        tet_mesh=tet_mesh,
        material_density=material_density,
        gravity=gravity,
        support_thickness=support_thickness,
    )
    free = ~fixed_mask
    u = np.zeros(total_dofs, dtype=np.float64)
    u[free] = spsolve(K[free][:, free], F[free])

    stress = np.zeros((len(elems), 6), dtype=np.float64)
    vm = np.zeros(len(elems), dtype=np.float64)
    for ei, tet in enumerate(elems):
        if not valid_elem[ei]:
            continue
        dofs = np.repeat(3 * tet, 3) + np.tile(np.arange(3), 4)
        strain = elem_B[ei] @ u[dofs]
        sigma = D @ strain
        stress[ei] = sigma
        sxx, syy, szz, syz, szx, sxy = sigma
        vm[ei] = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (syz**2 + szx**2 + sxy**2)
        )

    sf = yield_stress / max(vm.max(), 1e-12)
    disp = u.reshape(-1, 3)

    return {
        "nodes": nodes,
        "elems": elems,
        "displacement": disp,
        "stress": stress,
        "von_mises": vm,
        "sf_min": float(sf),
        "vm_max": float(vm.max()) if len(vm) else 0.0,
        "vm_mean": float(vm.mean()) if len(vm) else 0.0,
        "disp_max": float(np.abs(disp).max()) if len(disp) else 0.0,
        "n_nodes": int(len(nodes)),
        "n_elems": int(len(elems)),
        "n_valid_elems": int(valid_elem.sum()),
        "n_degenerate_elems": int((~valid_elem).sum()),
        "n_fixed_nodes": int(fixed_mask.reshape(-1, 3)[:, 0].sum()),
    }


def save_mesh_fem_result(output_dir: str | Path, result: Dict[str, object], source_obj: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_obj": str(source_obj),
        "mesh": {
            "n_nodes": int(result["n_nodes"]),
            "n_elems": int(result["n_elems"]),
            "n_valid_elems": int(result.get("n_valid_elems", result["n_elems"])),
            "n_degenerate_elems": int(result.get("n_degenerate_elems", 0)),
            "n_fixed_nodes": int(result["n_fixed_nodes"]),
        },
        "fem": {
            "sf_min": float(result["sf_min"]),
            "vm_max": float(result["vm_max"]),
            "vm_mean": float(result["vm_mean"]),
            "disp_max": float(result["disp_max"]),
        },
    }
    (out_dir / "mesh_fem_verification.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "mesh_nodes.npy", result["nodes"])
    np.save(out_dir / "mesh_elems.npy", result["elems"])
    np.save(out_dir / "mesh_displacement.npy", result["displacement"])
    np.save(out_dir / "mesh_stress.npy", result["stress"])
    np.save(out_dir / "mesh_von_mises.npy", result["von_mises"])
