"""
Voxel-based FEM solver for 3D linear elasticity (64³ scale)

Uses 8-node hexahedral elements (one per solid voxel).
Sparse assembly + direct solve via scipy.sparse.linalg.

Output: displacement field, stress field, Von Mises stress, true safety factor.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional
import time


def hex8_stiffness(E: float, nu: float, h: float = 1.0) -> np.ndarray:
    """Precompute 24×24 stiffness matrix for a unit hexahedral element.

    Uses analytical integration (2×2×2 Gauss quadrature) for isotropic
    linear elastic material with Young's modulus E and Poisson's ratio nu.
    Element side length h.
    """
    # Material matrix (6×6 for 3D isotropic)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    # 2×2×2 Gauss points
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([
        [-gp, -gp, -gp], [gp, -gp, -gp], [-gp, gp, -gp], [gp, gp, -gp],
        [-gp, -gp, gp],  [gp, -gp, gp],  [-gp, gp, gp],  [gp, gp, gp],
    ])

    # Node coordinates in natural coords [-1,1]³
    # Node ordering: (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)
    node_nat = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],
    ], dtype=float)

    Ke = np.zeros((24, 24))

    for gpt in gauss_pts:
        xi, eta, zeta = gpt

        # Shape function derivatives w.r.t. natural coordinates
        dN_dnat = np.zeros((3, 8))
        for i in range(8):
            xi_i, eta_i, zeta_i = node_nat[i]
            dN_dnat[0, i] = xi_i * (1 + eta_i * eta) * (1 + zeta_i * zeta) / 8.0
            dN_dnat[1, i] = (1 + xi_i * xi) * eta_i * (1 + zeta_i * zeta) / 8.0
            dN_dnat[2, i] = (1 + xi_i * xi) * (1 + eta_i * eta) * zeta_i / 8.0

        # Jacobian for uniform cube of side h
        J = dN_dnat @ (node_nat * h / 2.0)
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # Shape function derivatives w.r.t. physical coordinates
        dN_dx = invJ @ dN_dnat  # (3, 8)

        # B matrix (6×24)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i]   = dN_dx[0, i]  # εxx
            B[1, 3*i+1] = dN_dx[1, i]  # εyy
            B[2, 3*i+2] = dN_dx[2, i]  # εzz
            B[3, 3*i+1] = dN_dx[2, i]  # γyz
            B[3, 3*i+2] = dN_dx[1, i]
            B[4, 3*i]   = dN_dx[2, i]  # γzx
            B[4, 3*i+2] = dN_dx[0, i]
            B[5, 3*i]   = dN_dx[1, i]  # γxy
            B[5, 3*i+1] = dN_dx[0, i]

        Ke += B.T @ D @ B * detJ  # weight = 1 for each Gauss point

    return Ke


def voxel_fem_solve(
    density: np.ndarray,
    force_field: np.ndarray,
    fixed_mask: np.ndarray,
    E: float = 3.5e9,
    nu: float = 0.36,
    cell_size: float = 1.0,
    density_threshold: float = 0.3,
    return_nodal_displacement: bool = False,
    solver: str = "auto",
    iterative_rtol: float = 1e-8,
    iterative_maxiter: int = 4000,
    verbose: bool = True,
) -> dict:
    """Solve 3D linear elasticity on a voxel grid.

    Args:
        density: [N, N, N] voxel density (0~1)
        force_field: [3, N, N, N] body force per voxel
        fixed_mask: [1, N, N, N] or [N, N, N] fixed boundary (1=fixed)
        E: Young's modulus (Pa), default PLA
        nu: Poisson's ratio
        cell_size: voxel edge length
        density_threshold: minimum density to be considered solid
        verbose: print progress

    Returns:
        dict with:
          'displacement': [3, N, N, N]
          'stress': [6, N, N, N] (Voigt: sxx, syy, szz, syz, szx, sxy)
          'von_mises': [N, N, N]
          'sf_min': scalar safety factor (yield / max_vm in solid region)
          'n_elements': number of active elements
          'n_dofs': number of DOFs
          'solve_time_s': solve time
          'nodal_displacement': optional [3, N+1, N+1, N+1] FEM nodal field
          'nodal_active_mask': optional [N+1, N+1, N+1] active-node mask
    """
    N = density.shape[0]
    assert density.shape == (N, N, N)

    if fixed_mask.ndim == 4:
        fixed_mask = fixed_mask[0]

    t0 = time.time()

    # Identify solid voxels
    solid = density > density_threshold
    n_elements_total = int(solid.sum())

    if n_elements_total == 0:
        if verbose:
            print("No solid elements, returning zeros")
        return {
            'displacement': np.zeros((3, N, N, N), dtype=np.float32),
            'stress': np.zeros((6, N, N, N), dtype=np.float32),
            'von_mises': np.zeros((N, N, N), dtype=np.float32),
            'sf_min': 0.0,
            'n_elements': 0,
            'n_dofs': 0,
            'solve_time_s': 0.0,
        }

    # Find connected components — only solve for the component connected to
    # the fixed boundary to avoid singular matrix from floating elements
    from scipy.ndimage import label as ndlabel
    labeled, n_components = ndlabel(solid)

    # Identify which component(s) contain fixed boundary voxels
    fixed_voxels_idx = np.argwhere(fixed_mask > 0.5)
    connected_labels = set()
    for fv in fixed_voxels_idx:
        fi, fj, fk = fv[0], fv[1], fv[2]
        if solid[fi, fj, fk]:
            connected_labels.add(labeled[fi, fj, fk])
        # Also check neighbors (fixed voxel might be at surface)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    ni, nj, nk = fi+di, fj+dj, fk+dk
                    if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                        if solid[ni, nj, nk]:
                            connected_labels.add(labeled[ni, nj, nk])

    if not connected_labels:
        if verbose:
            print(f"WARNING: No solid voxels connected to fixed boundary!")
            print(f"  Total solid elements: {n_elements_total}, components: {n_components}")
        return {
            'displacement': np.zeros((3, N, N, N), dtype=np.float32),
            'stress': np.zeros((6, N, N, N), dtype=np.float32),
            'von_mises': np.zeros((N, N, N), dtype=np.float32),
            'sf_min': 0.0,
            'n_elements': n_elements_total,
            'n_dofs': 0,
            'solve_time_s': 0.0,
        }

    # Keep only the connected component(s)
    connected_mask = np.zeros_like(solid)
    for lbl in connected_labels:
        connected_mask |= (labeled == lbl)

    solid = solid & connected_mask
    n_elements = int(solid.sum())

    if verbose:
        print(f"FEM: N={N}, total solid={n_elements_total}, connected={n_elements} "
              f"({n_elements/(N**3)*100:.1f}%), components={n_components}, "
              f"connected_components={len(connected_labels)}")

    Nn = N + 1  # nodes per side

    # Precompute element stiffness
    Ke = hex8_stiffness(E, nu, cell_size)

    # 8 corner offsets for hex element at voxel (i,j,k)
    corner_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ])

    solid_voxels = np.argwhere(solid)  # (n_elements, 3)
    vi = solid_voxels[:, 0]
    vj = solid_voxels[:, 1]
    vk = solid_voxels[:, 2]

    # Compute global node indices (in full grid): (n_elements, 8)
    full_node_ids = np.zeros((n_elements, 8), dtype=np.int64)
    for c in range(8):
        ni = vi + corner_offsets[c, 0]
        nj = vj + corner_offsets[c, 1]
        nk = vk + corner_offsets[c, 2]
        full_node_ids[:, c] = ni * Nn * Nn + nj * Nn + nk

    # Reduced numbering: only include nodes used by active elements
    active_nodes = np.unique(full_node_ids.ravel())
    n_active_nodes = len(active_nodes)
    # Map: full_grid_node_id → reduced_id
    node_map = np.full(Nn ** 3, -1, dtype=np.int64)
    node_map[active_nodes] = np.arange(n_active_nodes)

    total_dofs = n_active_nodes * 3

    if verbose:
        print(f"FEM: active nodes={n_active_nodes}, total DOFs={total_dofs}")

    # Remap element node ids to reduced numbering
    node_ids = node_map[full_node_ids]  # (n_elements, 8)

    # Global DOF indices in reduced system: (n_elements, 24)
    elem_dofs = np.zeros((n_elements, 24), dtype=np.int64)
    for c in range(8):
        elem_dofs[:, 3*c]   = 3 * node_ids[:, c]
        elem_dofs[:, 3*c+1] = 3 * node_ids[:, c] + 1
        elem_dofs[:, 3*c+2] = 3 * node_ids[:, c] + 2

    # SIMP density weights
    rho_vals = density[vi, vj, vk]
    weights = rho_vals ** 3

    # Vectorized sparse assembly
    if verbose:
        print("FEM: Assembling stiffness matrix (vectorized)...")

    ri = np.repeat(np.arange(24), 24)   # length 576
    ci = np.tile(np.arange(24), 24)

    rows = elem_dofs[:, ri].ravel()  # (n_elements * 576,)
    cols = elem_dofs[:, ci].ravel()
    Ke_flat = Ke[ri, ci]
    vals = (weights[:, None] * Ke_flat[None, :]).ravel()

    # Force vector
    F = np.zeros(total_dofs, dtype=np.float64)
    for c in range(8):
        for d in range(3):
            np.add.at(F, 3 * node_ids[:, c] + d, force_field[d, vi, vj, vk] / 8.0)

    K = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(total_dofs, total_dofs)).tocsr()
    # These assembly arrays dominate memory at 96^3 and are not needed after CSR conversion.
    del rows, cols, vals

    if verbose:
        print(f"FEM: K assembled, nnz={K.nnz}")

    # Identify fixed DOFs in reduced numbering
    fixed_voxels_arr = np.argwhere(fixed_mask > 0.5)
    fixed_node_full = set()
    if len(fixed_voxels_arr) > 0:
        fi, fj, fk = fixed_voxels_arr[:, 0], fixed_voxels_arr[:, 1], fixed_voxels_arr[:, 2]
        for c in range(8):
            ni = fi + corner_offsets[c, 0]
            nj = fj + corner_offsets[c, 1]
            nk = fk + corner_offsets[c, 2]
            valid = (ni >= 0) & (ni < Nn) & (nj >= 0) & (nj < Nn) & (nk >= 0) & (nk < Nn)
            gn = ni[valid] * Nn * Nn + nj[valid] * Nn + nk[valid]
            fixed_node_full.update(gn.tolist())

    # Only fix nodes that are actually in the active set
    fixed_reduced = []
    for fn in fixed_node_full:
        rn = node_map[fn]
        if rn >= 0:
            fixed_reduced.extend([3*rn, 3*rn+1, 3*rn+2])
    fixed_dofs = np.array(sorted(fixed_reduced), dtype=np.int64)

    if verbose:
        print(f"FEM: {len(fixed_dofs)} fixed DOFs ({len(fixed_dofs)//3} fixed nodes)")

    if len(fixed_dofs) == 0:
        if verbose:
            print("WARNING: No fixed DOFs — system is unconstrained!")
        return {
            'displacement': np.zeros((3, N, N, N), dtype=np.float32),
            'stress': np.zeros((6, N, N, N), dtype=np.float32),
            'von_mises': np.zeros((N, N, N), dtype=np.float32),
            'sf_min': 0.0,
            'n_elements': n_elements,
            'n_dofs': total_dofs,
            'solve_time_s': time.time() - t0,
        }

    # Free DOFs
    free_mask_arr = np.ones(total_dofs, dtype=bool)
    free_mask_arr[fixed_dofs] = False
    free_dofs = np.where(free_mask_arr)[0]
    n_free = len(free_dofs)

    if verbose:
        print(f"FEM: Solving {n_free} free DOFs...")

    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]

    t_solve = time.time()
    use_iterative = solver == "iterative" or (solver == "auto" and n_free > 80000)
    if use_iterative:
        # AMG-preconditioned CG avoids the high memory cost of sparse LU at 96^3+.
        from scipy.sparse.linalg import cg
        try:
            import pyamg
        except ImportError as exc:
            raise RuntimeError("iterative FEM requires pyamg") from exc
        if verbose:
            print(f"FEM: Using AMG-preconditioned CG (n_free={n_free})")
        ml = pyamg.smoothed_aggregation_solver(K_free, symmetry="symmetric")
        M = ml.aspreconditioner(cycle="V")
        U_free, info = cg(K_free, F_free, M=M, rtol=iterative_rtol, atol=0.0,
                          maxiter=iterative_maxiter)
        if info != 0:
            raise RuntimeError(f"AMG-preconditioned CG failed to converge (info={info})")
    else:
        U_free = spsolve(K_free, F_free)
    solve_time = time.time() - t_solve

    if verbose:
        print(f"FEM: Solve completed in {solve_time:.2f}s")

    # Full displacement vector (in reduced numbering)
    U = np.zeros(total_dofs, dtype=np.float64)
    U[free_dofs] = U_free

    # Vectorized displacement/stress extraction
    disp_field = np.zeros((3, N, N, N), dtype=np.float32)
    stress_field = np.zeros((6, N, N, N), dtype=np.float32)

    B_center = _compute_B_center(cell_size)

    # elem_dofs already computed above: (n_elements, 24)
    # Gather all element displacements: (n_elements, 24)
    U_elems = U[elem_dofs]  # fancy indexing

    # Average nodal displacement per direction: (n_elements, 3)
    for d in range(3):
        avg_d = U_elems[:, d::3].mean(axis=1)  # (n_elements,)
        disp_field[d, solid_voxels[:, 0], solid_voxels[:, 1], solid_voxels[:, 2]] = avg_d.astype(np.float32)

    # Strain at element center: (n_elements, 6) = (n_elements, 24) @ (24, 6)
    strains = U_elems @ B_center.T  # (n_elements, 6)

    # Stress = D @ strain (material matrix)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    stresses = np.zeros_like(strains)
    trace = strains[:, 0] + strains[:, 1] + strains[:, 2]
    stresses[:, 0] = lam * trace + 2 * mu * strains[:, 0]
    stresses[:, 1] = lam * trace + 2 * mu * strains[:, 1]
    stresses[:, 2] = lam * trace + 2 * mu * strains[:, 2]
    stresses[:, 3] = mu * strains[:, 3]
    stresses[:, 4] = mu * strains[:, 4]
    stresses[:, 5] = mu * strains[:, 5]

    for s in range(6):
        stress_field[s, solid_voxels[:, 0], solid_voxels[:, 1], solid_voxels[:, 2]] = stresses[:, s].astype(np.float32)

    # Von Mises stress
    sxx = stress_field[0]
    syy = stress_field[1]
    szz = stress_field[2]
    syz = stress_field[3]
    szx = stress_field[4]
    sxy = stress_field[5]

    vm = np.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
        + 1e-30
    )

    # Safety factor: yield / max_vm in solid region (connected only)
    vm_solid = vm[solid]
    if len(vm_solid) > 0 and vm_solid.max() > 0:
        sf_min = 50.0e6 / vm_solid.max()
    else:
        sf_min = float('inf')

    # Also report for all solid voxels (including disconnected — those have zero stress)
    all_solid = density > density_threshold
    vm_all_solid = vm[all_solid]

    total_time = time.time() - t0

    if verbose:
        print(f"FEM: max displacement = {np.abs(disp_field).max():.6e}")
        print(f"FEM: max Von Mises (solid) = {vm_solid.max():.6e} Pa")
        print(f"FEM: SF_min = {sf_min:.4f}")
        print(f"FEM: Total time = {total_time:.2f}s")

    result = {
        'displacement': disp_field,
        'stress': stress_field,
        'von_mises': vm,
        'sf_min': sf_min,
        'vm_max': float(vm_solid.max()) if len(vm_solid) > 0 else 0.0,
        'n_elements': n_elements,
        'n_elements_total': n_elements_total,
        'n_disconnected': n_elements_total - n_elements,
        'n_dofs': n_free,
        'solve_time_s': solve_time,
        'total_time_s': total_time,
    }
    if return_nodal_displacement:
        # Keep the exact Hex8 solution at grid nodes.  The existing
        # ``displacement`` output is an element-average field and must not be
        # used as a nodal target for a discrete equilibrium loss.
        nodal_displacement = np.zeros((3, Nn, Nn, Nn), dtype=np.float64)
        nodal_active_mask = np.zeros((Nn, Nn, Nn), dtype=np.uint8)
        ni = active_nodes // (Nn * Nn)
        rem = active_nodes % (Nn * Nn)
        nj = rem // Nn
        nk = rem % Nn
        nodal_active_mask[ni, nj, nk] = 1
        for d in range(3):
            nodal_displacement[d, ni, nj, nk] = U[d::3]
        result['nodal_displacement'] = nodal_displacement
        result['nodal_active_mask'] = nodal_active_mask
    return result


def _compute_B_center(h: float) -> np.ndarray:
    """Compute B matrix at element center (xi=eta=zeta=0)."""
    node_nat = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],
    ], dtype=float)

    # At center (0,0,0), dN/dxi = xi_i * (1)(1) / 8 etc.
    dN_dnat = np.zeros((3, 8))
    for i in range(8):
        xi_i, eta_i, zeta_i = node_nat[i]
        dN_dnat[0, i] = xi_i * 1.0 * 1.0 / 8.0
        dN_dnat[1, i] = 1.0 * eta_i * 1.0 / 8.0
        dN_dnat[2, i] = 1.0 * 1.0 * zeta_i / 8.0

    J = dN_dnat @ (node_nat * h / 2.0)
    invJ = np.linalg.inv(J)
    dN_dx = invJ @ dN_dnat

    B = np.zeros((6, 24))
    for i in range(8):
        B[0, 3*i]   = dN_dx[0, i]
        B[1, 3*i+1] = dN_dx[1, i]
        B[2, 3*i+2] = dN_dx[2, i]
        B[3, 3*i+1] = dN_dx[2, i]
        B[3, 3*i+2] = dN_dx[1, i]
        B[4, 3*i]   = dN_dx[2, i]
        B[4, 3*i+2] = dN_dx[0, i]
        B[5, 3*i]   = dN_dx[1, i]
        B[5, 3*i+1] = dN_dx[0, i]

    return B


def quick_sf_check(
    density: np.ndarray,
    E: float = 3.5e9,
    nu: float = 0.36,
    yield_stress: float = 50.0e6,
    material_density: float = 1.24e3,
    gravity: float = -9.81,
    verbose: bool = True,
) -> dict:
    """Quick safety factor check using standing load (gravity + bottom fixed).

    Convenience wrapper that generates standing load and runs FEM.
    """
    from structural.analysis.load_generator import LoadGenerator

    N = density.shape[0]
    load_gen = LoadGenerator(
        material_density=material_density,
        gravity=gravity,
    )
    load = load_gen.standing_load(density)

    result = voxel_fem_solve(
        density=density,
        force_field=load['force'],
        fixed_mask=load['fixed_mask'],
        E=E,
        nu=nu,
        density_threshold=0.3,
        verbose=verbose,
    )
    result['yield_stress'] = yield_stress

    # Recompute SF with given yield_stress
    solid = density > 0.3
    vm_solid = result['von_mises'][solid]
    if len(vm_solid) > 0 and vm_solid.max() > 0:
        result['sf_min'] = yield_stress / vm_solid.max()
    else:
        result['sf_min'] = float('inf')

    return result
