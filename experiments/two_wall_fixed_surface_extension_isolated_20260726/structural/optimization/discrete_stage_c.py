"""
Discrete topology optimization for Stage C — FEM-guided BESO approach.

The PINO surrogate's scale head is nearly invariant to density changes,
so we cannot use PINO alone for iterative optimization. Instead:

1. Use FEM to get ground-truth stress fields (expensive but accurate)
2. Apply evolutionary structural optimization (BESO) rules:
   - Add material at high-stress surface voxels
   - Remove material at low-stress surface voxels
3. Re-evaluate with FEM every few iterations to update stress field
4. Use PINO only for cheap intermediate screening

This trades PINO speed for FEM accuracy where it matters.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time


def get_surface_voxels(density: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Find surface voxel coordinates (interface between solid and void).

    Returns array of shape (N, 3) with ijk coordinates.
    """
    solid = (density > threshold).astype(np.float32)
    from scipy.ndimage import binary_dilation, binary_erosion
    dilated = binary_dilation(solid > 0.5, iterations=1).astype(np.float32)
    eroded = binary_erosion(solid > 0.5, iterations=1).astype(np.float32)
    surface = ((dilated - eroded) > 0.5)
    coords = np.argwhere(surface)
    return coords


def compute_silhouette_iou(density: np.ndarray, density_original: np.ndarray,
                           threshold: float = 0.5) -> float:
    """Compute IoU between two binary voxel fields projected along 3 axes."""
    solid_new = density > threshold
    solid_orig = density_original > threshold
    ious = []
    for axis in range(3):
        proj_new = solid_new.any(axis=axis)
        proj_orig = solid_orig.any(axis=axis)
        intersection = (proj_new & proj_orig).sum()
        union = (proj_new | proj_orig).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0)
    return float(np.mean(ious))


def discrete_stage_c(
    pino,
    density_np: np.ndarray,
    force_tensor: torch.Tensor,
    device: torch.device,
    sf_target: float = 1.3,
    max_iters: int = 200,
    batch_size: int = 64,
    iou_threshold: float = 0.90,
    vol_budget: float = 1.3,
    output_dir: str | None = None,
    yield_stress: float = 250e6,
    fem_force: np.ndarray | None = None,
    fem_fixed_mask: np.ndarray | None = None,
    fem_interval: int = 5,
    calibrator=None,
    stress_source: str = "fem",
    density_reference_np: np.ndarray | None = None,
    enable_trim: bool = True,
    appearance_aware: bool = False,
):
    """FEM-guided discrete topology optimization.

    stress_source:
        "fem"  (default) — BESO driven by voxel_fem_solve; PINO only early-triggers.
                           Existing behavior, unchanged.
        "pino"           — BESO driven by PINO's predicted von Mises field
                           (forward_surrogate_physical). Much faster, but no
                           ground-truth anchor; caller should run a single FEM
                           verify on the returned density.

    Strategy (BESO-style):
    - Use FEM stress field to identify where to add/remove material
    - Phase 1 (reinforce): Add material at high-stress surface locations
    - Phase 2 (trim): Remove low-stress surface material
    - Re-run FEM every `fem_interval` iterations to update stress field
    - PINO used only for cheap initial assessment

    Args:
        pino: Loaded PINO model (for initial quick eval only)
        density_np: Binary density array [N, N, N] or [1, N, N, N]
        force_tensor: Force field [3, N, N, N]
        device: torch device
        sf_target: Target safety factor
        max_iters: Maximum iterations per phase
        batch_size: Voxels to flip per iteration
        iou_threshold: Minimum silhouette IoU
        vol_budget: Maximum volume ratio vs original
        output_dir: Where to save results
        yield_stress: Material yield stress (Pa)
        fem_force: Force field for FEM [3, N, N, N] numpy
        fem_fixed_mask: Fixed boundary mask [1, N, N, N] or [N, N, N] numpy
        fem_interval: Re-run FEM every N iterations
        calibrator: Optional CalibratedPINO instance. When provided, used as a
            cheap surrogate to predict log_vm_max between FEM calls and adapt
            the FEM check cadence (early-trigger when SF drift is large).
        density_reference_np: Optional original/reference density used for IoU
            and volume-ratio constraints. When omitted, the starting density is
            treated as the reference. This is useful for continuation runs that
            start from an already-repaired density but should still preserve the
            original silhouette/volume contract.
        enable_trim: Whether to run the volume-reduction trim phase after the
            reinforce phase. Defaults to True for backwards compatibility.
        appearance_aware: When True, reinforce candidate ranking first prefers
            additions that do not expand any of the three reference silhouettes,
            then breaks ties by predicted stress. Defaults to False to preserve
            existing fixed-policy behavior.
    """
    from structural.analysis.fem_solver import voxel_fem_solve
    if stress_source not in ("fem", "pino"):
        raise ValueError(f"stress_source must be 'fem' or 'pino', got {stress_source!r}")
    if stress_source == "pino":
        from structural.models.load_model import forward_surrogate_physical

    def cal_sf(d):
        """Cheap calibrator-based safety factor estimate."""
        if calibrator is None:
            return None
        try:
            log_vm = calibrator.predict_log_vm_max(d, force=fem_force)
            return float(yield_stress / (10.0 ** log_vm))
        except Exception:
            return None

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    # Normalize density
    if density_np.ndim == 4:
        density_np = density_np[0]
    density = (density_np > 0.5).astype(np.float32)
    if density_reference_np is not None:
        if density_reference_np.ndim == 4:
            density_reference_np = density_reference_np[0]
        density_original = (density_reference_np > 0.5).astype(np.float32)
    else:
        density_original = density.copy()
    N = density.shape[0]

    # Prepare FEM inputs
    if fem_force is None:
        fem_force = force_tensor.cpu().numpy()
    if fem_fixed_mask is None:
        # Default: fix bottom layer
        fem_fixed_mask = np.zeros((N, N, N), dtype=np.float32)
        fem_fixed_mask[:, 0, :] = 1.0  # fix y=0 layer
        print("  Warning: No fixed_mask provided, using default (y=0 fixed)")

    vol_init = density_original.sum()
    ref_proj0 = (density_original > 0.5).any(axis=0)
    ref_proj1 = (density_original > 0.5).any(axis=1)
    ref_proj2 = (density_original > 0.5).any(axis=2)

    def silhouette_expansion_cost(i: int, j: int, k: int) -> int:
        return int(not ref_proj0[j, k]) + int(not ref_proj1[i, k]) + int(not ref_proj2[i, j])

    def _run_fem_real(d):
        result = voxel_fem_solve(
            d, fem_force, fem_fixed_mask,
            density_threshold=0.3, verbose=False,
        )
        vm_field = result['von_mises']
        solid = d > 0.3
        vm_solid = vm_field[solid]
        vm_max = float(vm_solid.max()) if vm_solid.size > 0 else 1e-12
        sf = yield_stress / (vm_max + 1e-12)
        u_max = float(np.abs(result['displacement']).max())
        return sf, vm_max, vm_field, u_max

    def _run_pino_field(d):
        """Pure-PINO stress evaluation.

        vm_field: uncalibrated PINO von Mises (used only to rank flip candidates;
                  direction, not magnitude).
        sf:       isotonic-calibrated scalar SF from `calibrator` if provided
                  (matches the detection-stage SF estimate; reliable for stop
                  condition). Falls back to yield/max(vm) when no calibrator.
        """
        d_t = torch.from_numpy(d.astype(np.float32)).to(device)
        # build_pino_input expects density [B,1,N,N,N] and force [B,3,N,N,N]
        if d_t.ndim == 3:
            d_t = d_t.unsqueeze(0).unsqueeze(0)
        elif d_t.ndim == 4:
            d_t = d_t.unsqueeze(0)
        f_t = force_tensor.to(device)
        if f_t.ndim == 4:
            f_t = f_t.unsqueeze(0)
        with torch.no_grad():
            phys = forward_surrogate_physical(pino, d_t, f_t)
        vm = phys['von_mises'].squeeze().detach().cpu().numpy().astype(np.float32)
        disp = phys['displacement'].detach().cpu().numpy()
        solid = d > 0.3
        vm_solid = vm[solid]
        vm_max = float(vm_solid.max()) if vm_solid.size > 0 else 1e-12
        u_max = float(np.abs(disp).max())
        # Calibrated scalar SF — isotonic-calibrated on detection hardset,
        # robust to miscalibration of raw field magnitude. Falls back to raw.
        sf_cal = cal_sf(d)
        sf = sf_cal if sf_cal is not None else yield_stress / (vm_max + 1e-12)
        return sf, vm_max, vm, u_max

    def run_fem(d):
        """Return (sf, vm_max, vm_field, u_max) from the configured stress source."""
        if stress_source == "pino":
            return _run_pino_field(d)
        return _run_fem_real(d)

    # Initial FEM evaluation
    print("  [Stage C] Running initial FEM evaluation...")
    t0 = time.time()
    sf_init, vm_max_init, vm_field, u_max_init = run_fem(density)
    t_fem = time.time() - t0
    print(f"  [Stage C] Initial: SF={sf_init:.3f}, VM_max={vm_max_init:.2e}, "
          f"u_max={u_max_init:.4e}, vol={vol_init:.0f} (FEM took {t_fem:.1f}s)")

    history = [{
        'iter': 0, 'sf': sf_init, 'vm_max': vm_max_init, 'u_max': u_max_init,
        'vol_ratio': 1.0, 'iou': 1.0, 'n_flips': 0, 'phase': 'init',
        'fem_eval': True,
    }]

    best_density = density.copy()
    best_sf = sf_init
    best_feasible_density = None
    best_feasible_sf = 0.0
    sf_current = sf_init

    if sf_init >= sf_target:
        best_feasible_density = density.copy()
        best_feasible_sf = sf_init

    # ========== Phase 1: Reinforce ==========
    if sf_current < sf_target:
        print(f"\n  === Phase 1: Reinforce (target SF >= {sf_target}) ===")
        stagnation_count = 0

        for it in tqdm(range(1, max_iters + 1), desc="Phase 1 (reinforce)"):
            if sf_current >= sf_target:
                print(f"\n  SF target reached at iter {it}: SF={sf_current:.3f}")
                break

            # Find void voxels adjacent to high-stress solid voxels
            surface = get_surface_voxels(density)
            if len(surface) == 0:
                break

            void_candidates = []
            for coord in surface:
                i, j, k = coord
                if density[i, j, k] < 0.5:
                    i_lo, i_hi = max(0, i-1), min(N, i+2)
                    j_lo, j_hi = max(0, j-1), min(N, j+2)
                    k_lo, k_hi = max(0, k-1), min(N, k+2)
                    neighbor_stress = vm_field[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]
                    solid_neighbor = density[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] > 0.5
                    if solid_neighbor.any():
                        score = float(neighbor_stress[solid_neighbor].max())
                        visual_cost = silhouette_expansion_cost(i, j, k) if appearance_aware else 0
                        void_candidates.append((i, j, k, score, visual_cost))

            if not void_candidates:
                break

            # Sort by stress (highest first)
            if appearance_aware:
                void_candidates.sort(key=lambda x: (x[4], -x[3]))
            else:
                void_candidates.sort(key=lambda x: -x[3])
            n_flip = min(batch_size, len(void_candidates))
            candidates = void_candidates[:n_flip]

            # Apply flips
            density_new = density.copy()
            for i, j, k, *_ in candidates:
                density_new[i, j, k] = 1.0

            # Check appearance constraint before expensive FEM
            iou = compute_silhouette_iou(density_new, density_original)
            vol_ratio = density_new.sum() / max(vol_init, 1)
            if iou < iou_threshold or vol_ratio > vol_budget:
                # Try smaller batch
                n_flip = max(1, n_flip // 4)
                candidates = void_candidates[:n_flip]
                density_new = density.copy()
                for i, j, k, *_ in candidates:
                    density_new[i, j, k] = 1.0
                iou = compute_silhouette_iou(density_new, density_original)
                vol_ratio = density_new.sum() / max(vol_init, 1)
                if iou < iou_threshold or vol_ratio > vol_budget:
                    stagnation_count += 1
                    if stagnation_count > 10:
                        print(f"\n  Phase 1 stagnated (constraints) at iter {it}")
                        break
                    continue

            # Accept the flip (stress-guided, verified periodically by FEM)
            density = density_new
            stagnation_count = 0

            # Decide when to call FEM. Default: fixed interval.
            # If a calibrator is supplied, also early-trigger when the cheap
            # surrogate suggests we may have crossed the SF target — saves
            # iterations vs waiting fem_interval steps.
            need_fem = (it % fem_interval == 0) or (it == 1)
            if calibrator is not None and not need_fem:
                cal_sf_val = cal_sf(density)
                if cal_sf_val is not None and cal_sf_val >= sf_target * 0.95:
                    need_fem = True
            if need_fem:
                sf_new, vm_max_new, vm_field, u_max_new = run_fem(density)
                sf_current = sf_new
                print(f"\n  [FEM @iter {it}] SF={sf_new:.3f}, VM_max={vm_max_new:.2e}, "
                      f"vol_ratio={vol_ratio:.3f}")

            history.append({
                'iter': it, 'sf': sf_current,
                'vol_ratio': float(density.sum() / max(vol_init, 1)),
                'iou': float(iou), 'n_flips': n_flip, 'phase': 'reinforce',
                'fem_eval': need_fem,
            })

            if sf_current >= sf_target:
                best_feasible_density = density.copy()
                best_feasible_sf = sf_current

        # Final FEM check after Phase 1
        sf_current, vm_max_current, vm_field, u_max_current = run_fem(density)
        print(f"  Phase 1 done: SF={sf_current:.3f}, vol_ratio={density.sum()/vol_init:.3f}")

        if sf_current >= sf_target:
            best_feasible_density = density.copy()
            best_feasible_sf = sf_current

    # ========== Phase 2: Trim ==========
    if enable_trim and sf_current >= sf_target:
        print(f"\n  === Phase 2: Trim (minimize volume, keep SF >= {sf_target}) ===")
        stagnation_count = 0

        for it in tqdm(range(1, max_iters + 1), desc="Phase 2 (trim)"):
            surface = get_surface_voxels(density)
            if len(surface) == 0:
                break

            # Find solid surface voxels with LOW stress
            solid_candidates = []
            for coord in surface:
                i, j, k = coord
                if density[i, j, k] > 0.5:
                    stress = vm_field[i, j, k]
                    solid_candidates.append((i, j, k, stress))

            if not solid_candidates:
                break

            solid_candidates.sort(key=lambda x: x[3])
            n_flip = min(batch_size // 2, len(solid_candidates))
            candidates = solid_candidates[:n_flip]

            density_new = density.copy()
            for i, j, k, _ in candidates:
                density_new[i, j, k] = 0.0

            iou = compute_silhouette_iou(density_new, density_original)
            if iou < iou_threshold:
                n_flip = max(1, n_flip // 4)
                candidates = solid_candidates[:n_flip]
                density_new = density.copy()
                for i, j, k, _ in candidates:
                    density_new[i, j, k] = 0.0
                iou = compute_silhouette_iou(density_new, density_original)
                if iou < iou_threshold:
                    stagnation_count += 1
                    if stagnation_count > 10:
                        break
                    continue

            # Run FEM periodically to verify SF stays above target
            need_fem = (it % fem_interval == 0) or (it == 1)
            if need_fem:
                sf_new, vm_max_new, vm_field_new, u_max_new = run_fem(density_new)
                if sf_new >= sf_target:
                    density = density_new
                    sf_current = sf_new
                    vm_field = vm_field_new
                    stagnation_count = 0
                    best_feasible_density = density.copy()
                    best_feasible_sf = sf_current
                else:
                    # FEM says unsafe — try smaller batch
                    n_flip = max(1, n_flip // 4)
                    candidates = solid_candidates[:n_flip]
                    density_new = density.copy()
                    for i, j, k, _ in candidates:
                        density_new[i, j, k] = 0.0
                    sf_new2, _, vm_field_new2, _ = run_fem(density_new)
                    if sf_new2 >= sf_target:
                        density = density_new
                        sf_current = sf_new2
                        vm_field = vm_field_new2
                        stagnation_count = 0
                        best_feasible_density = density.copy()
                        best_feasible_sf = sf_current
                    else:
                        stagnation_count += 1
            else:
                # Accept optimistically between FEM checks
                density = density_new
                stagnation_count = 0

            history.append({
                'iter': max_iters + it, 'sf': sf_current,
                'vol_ratio': float(density.sum() / max(vol_init, 1)),
                'iou': float(compute_silhouette_iou(density, density_original)),
                'n_flips': n_flip, 'phase': 'trim',
                'fem_eval': need_fem,
            })

            if stagnation_count > 10:
                print(f"\n  Phase 2 stagnated at iter {it}")
                break

        # Final FEM verification after trim
        sf_current, vm_max_current, vm_field, u_max_current = run_fem(density)
        if sf_current < sf_target and best_feasible_density is not None:
            print(f"  Trim went too far (SF={sf_current:.3f}), reverting to best feasible")
            density = best_feasible_density.copy()
            sf_current = best_feasible_sf

    # ========== Final results ==========
    sf_final, vm_max_final, _, u_max_final = run_fem(density)
    iou_final = compute_silhouette_iou(density, density_original)
    vol_ratio_final = float(density.sum() / max(vol_init, 1))

    result = {
        'sf_initial': sf_init,
        'sf_final': sf_final,
        'sf_target': sf_target,
        'sf_target_reached': sf_final >= sf_target,
        'vol_ratio_final': vol_ratio_final,
        'iou_final': iou_final,
        'u_max_final': u_max_final,
        'vm_max_final': vm_max_final,
        'n_iters': len(history),
        'n_voxels_changed': int(np.abs(density - density_original).sum()),
    }

    if best_feasible_density is not None:
        result['best_feasible_sf'] = best_feasible_sf

    print(f"\n  [Stage C] Final: SF={sf_final:.3f} (target={sf_target}), "
          f"vol_ratio={vol_ratio_final:.3f}, IoU={iou_final:.3f}, "
          f"voxels_changed={result['n_voxels_changed']}")

    # Save
    if output_dir:
        out_path = Path(output_dir)
        np.save(out_path / 'final_voxels.npy', density)
        np.save(out_path / 'original_voxels.npy', density_original)
        with open(out_path / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        with open(out_path / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)
        if best_feasible_density is not None:
            np.save(out_path / 'best_feasible_voxels.npy', best_feasible_density)

    return result, density


def discrete_stage_c_v2(
    pino,
    density_np: np.ndarray,
    force_tensor: torch.Tensor,
    device: torch.device,
    sf_target: float = 1.3,
    max_iters: int = 8,
    batch_size: int = 64,
    iou_threshold: float = 0.90,
    vol_budget: float = 1.3,
    output_dir: str | None = None,
    yield_stress: float = 250e6,
    fem_force: np.ndarray | None = None,
    fem_fixed_mask: np.ndarray | None = None,
    fem_interval: int = 2,
    calibrator=None,
    topk_window: int = 256,
    softmax_temp_frac: float = 0.1,
    plateau_strikes_max: int = 2,
    perturb_voxels: int = 32,
    rng_seed: int | None = 0,
):
    """Phase-1-only repair variant with (a) SF-verified rollback to best_density,
    (b) softmax-weighted top-K candidate sampling to break out of argmax hot-spot
    cycling, and (c) perturbation restart from best_density after consecutive
    non-improving FEM checks. Falls through to original Phase-2 trim once SF >=
    target.
    """
    from structural.analysis.fem_solver import voxel_fem_solve

    rng = np.random.default_rng(rng_seed)

    if density_np.ndim == 4:
        density_np = density_np[0]
    density = (density_np > 0.5).astype(np.float32)
    density_original = density.copy()
    N = density.shape[0]

    if fem_force is None:
        fem_force = force_tensor.cpu().numpy()
    if fem_fixed_mask is None:
        fem_fixed_mask = np.zeros((N, N, N), dtype=np.float32)
        fem_fixed_mask[:, 0, :] = 1.0

    vol_init = density.sum()

    def run_fem(d):
        result = voxel_fem_solve(d, fem_force, fem_fixed_mask,
                                 density_threshold=0.3, verbose=False)
        vm_field = result['von_mises']
        solid = d > 0.3
        vm_solid = vm_field[solid]
        vm_max = float(vm_solid.max()) if vm_solid.size > 0 else 1e-12
        sf = yield_stress / (vm_max + 1e-12)
        u_max = float(np.abs(result['displacement']).max())
        return sf, vm_max, vm_field, u_max

    print("  [Stage C v2] Running initial FEM evaluation...")
    t0 = time.time()
    sf_init, vm_max_init, vm_field, u_max_init = run_fem(density)
    t_fem = time.time() - t0
    print(f"  [Stage C v2] Initial: SF={sf_init:.3f}, VM_max={vm_max_init:.2e}, "
          f"vol={vol_init:.0f} (FEM took {t_fem:.1f}s)")

    history = [{
        'iter': 0, 'sf': sf_init, 'vm_max': vm_max_init, 'u_max': u_max_init,
        'vol_ratio': 1.0, 'iou': 1.0, 'n_flips': 0, 'phase': 'init',
        'fem_eval': True,
    }]

    best_density = density.copy()
    best_sf = sf_init
    sf_current = sf_init
    best_feasible_density = density.copy() if sf_init >= sf_target else None
    best_feasible_sf = sf_init if sf_init >= sf_target else 0.0

    if sf_current < sf_target:
        print(f"\n  === Phase 1 v2: Reinforce (target SF >= {sf_target}) ===")
        plateau_strikes = 0

        for it in tqdm(range(1, max_iters + 1), desc="Phase 1 v2"):
            if sf_current >= sf_target:
                print(f"\n  [v2] target reached at iter {it}: SF={sf_current:.3f}")
                break

            surface = get_surface_voxels(density)
            if len(surface) == 0:
                break

            void_candidates = []
            for coord in surface:
                i, j, k = coord
                if density[i, j, k] < 0.5:
                    i_lo, i_hi = max(0, i-1), min(N, i+2)
                    j_lo, j_hi = max(0, j-1), min(N, j+2)
                    k_lo, k_hi = max(0, k-1), min(N, k+2)
                    solid_nb = density[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi] > 0.5
                    if solid_nb.any():
                        neigh_stress = vm_field[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]
                        score = float(neigh_stress[solid_nb].max())
                        void_candidates.append((i, j, k, score))

            if not void_candidates:
                break

            void_candidates.sort(key=lambda x: -x[3])
            K = min(topk_window, len(void_candidates))
            top_k = void_candidates[:K]
            scores = np.array([c[3] for c in top_k], dtype=np.float64)
            smax = scores.max()
            temp = max(smax * softmax_temp_frac, 1e-9)
            logits = (scores - smax) / temp
            probs = np.exp(logits)
            probs /= probs.sum()

            n_flip = min(batch_size, K)
            idx = rng.choice(K, size=n_flip, replace=False, p=probs)
            chosen = [top_k[i] for i in idx]

            density_new = density.copy()
            for i, j, k, _ in chosen:
                density_new[i, j, k] = 1.0

            iou = compute_silhouette_iou(density_new, density_original)
            vol_ratio = density_new.sum() / max(vol_init, 1)
            if iou < iou_threshold or vol_ratio > vol_budget:
                n_flip = max(1, n_flip // 4)
                idx = rng.choice(K, size=n_flip, replace=False,
                                 p=probs if n_flip <= K else None)
                chosen = [top_k[i] for i in idx]
                density_new = density.copy()
                for i, j, k, _ in chosen:
                    density_new[i, j, k] = 1.0
                iou = compute_silhouette_iou(density_new, density_original)
                vol_ratio = density_new.sum() / max(vol_init, 1)
                if iou < iou_threshold or vol_ratio > vol_budget:
                    continue

            density = density_new

            need_fem = (it % fem_interval == 0) or (it == 1)
            if need_fem:
                sf_new, vm_max_new, vm_field_new, u_max_new = run_fem(density)
                print(f"\n  [v2 FEM @iter {it}] SF={sf_new:.3f} (best {best_sf:.3f})")

                if sf_new > best_sf + 0.005:
                    best_sf = sf_new
                    best_density = density.copy()
                    vm_field = vm_field_new
                    sf_current = sf_new
                    plateau_strikes = 0
                    if sf_new >= sf_target:
                        best_feasible_density = density.copy()
                        best_feasible_sf = sf_new
                else:
                    density = best_density.copy()
                    sf_current = best_sf
                    plateau_strikes += 1
                    print(f"  [v2] rollback; plateau_strikes={plateau_strikes}")

                    if plateau_strikes >= plateau_strikes_max:
                        print(f"  [v2] perturbation restart at iter {it}")
                        density = best_density.copy()
                        void_coords = np.argwhere(density < 0.5)
                        if len(void_coords) >= perturb_voxels:
                            pick = rng.choice(len(void_coords),
                                              size=perturb_voxels, replace=False)
                            for p in pick:
                                i, j, k = void_coords[p]
                                density[i, j, k] = 1.0
                        sf_p, vm_max_p, vm_field_p, u_max_p = run_fem(density)
                        print(f"  [v2] post-perturb SF={sf_p:.3f}")
                        if sf_p > best_sf:
                            best_sf = sf_p
                            best_density = density.copy()
                        else:
                            density = best_density.copy()
                        vm_field = vm_field_p
                        sf_current = sf_p if sf_p > best_sf else best_sf
                        plateau_strikes = 0

            history.append({
                'iter': it, 'sf': sf_current,
                'vol_ratio': float(density.sum() / max(vol_init, 1)),
                'iou': float(iou), 'n_flips': n_flip, 'phase': 'reinforce_v2',
                'fem_eval': need_fem, 'plateau_strikes': plateau_strikes,
            })

        density = best_density.copy()
        sf_current, vm_max_current, vm_field, u_max_current = run_fem(density)
        print(f"  Phase 1 v2 done: SF={sf_current:.3f} (best ever {best_sf:.3f})")

        if sf_current >= sf_target:
            best_feasible_density = density.copy()
            best_feasible_sf = sf_current

    sf_final, vm_max_final, _, u_max_final = run_fem(density)
    iou_final = compute_silhouette_iou(density, density_original)
    vol_ratio_final = float(density.sum() / max(vol_init, 1))

    result = {
        'sf_initial': sf_init,
        'sf_final': sf_final,
        'sf_target': sf_target,
        'sf_target_reached': sf_final >= sf_target,
        'vol_ratio_final': vol_ratio_final,
        'iou_final': iou_final,
        'u_max_final': u_max_final,
        'vm_max_final': vm_max_final,
        'n_iters': len(history),
        'n_voxels_changed': int(np.abs(density - density_original).sum()),
        'best_sf_ever': best_sf,
    }

    print(f"\n  [Stage C v2] Final: SF={sf_final:.3f} (target={sf_target}), "
          f"vol_ratio={vol_ratio_final:.3f}, IoU={iou_final:.3f}")

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / 'final_voxels.npy', density)
        np.save(out_path / 'original_voxels.npy', density_original)
        with open(out_path / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        with open(out_path / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)

    return result, density
