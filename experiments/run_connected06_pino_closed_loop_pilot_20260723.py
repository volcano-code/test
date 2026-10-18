#!/usr/bin/env python3
"""Single-case pilot for PINO-guided, closed-loop material reduction.

The experiment starts from the exact projection-preserving fixed point of the
production geometry trajectory.  It then relaxes only the projection-retention
budget and compares soft-only deletion ordering with PINO stress-aware ordering.
PINO is evaluated after every accepted batch; an unsafe prediction rolls the
batch back.  Voxel FEM is used only to audit the retained checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from candidate_selection_batch import fem_sf
from run_all_baselines import load_pino
from shadowforge_pipeline.run import (
    FEMDatasetV3,
    PINO_YIELD_STRESS,
    compute_von_mises,
)
from shadowforge_pipeline.unified_run import (
    S6,
    _generic_seed_trajectory,
    _generic_connected_delivery,
    _generic_metrics,
    _generic_targets,
    _generic_tfs_carve_trajectory,
    _generic_visual_hull,
    axis_guarded_carve_step,
    downsample_max_to,
    local_simple_point,
    print_orientations,
    structural_proxy,
)
from structural.analysis.load_generator import LoadGenerator


def normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[mask]
    if not selected.size:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.quantile(selected, [0.02, 0.98])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def orient_y_down(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (0, 2, 1))[:, :, ::-1].copy()


def undo_orient_y_down(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume[:, :, ::-1], (0, 2, 1)).copy()


@torch.no_grad()
def pino_state_oriented(
    model: torch.nn.Module,
    oriented_volume: np.ndarray,
    device: torch.device,
) -> dict:
    """Return scalar SF and an oriented physical von Mises field."""
    oriented = np.asarray(oriented_volume, dtype=np.float32)
    load = LoadGenerator().standing_load(oriented)
    density_np = oriented.astype(np.float32)
    force_np = load["force"].astype(np.float32)
    fixed_np = load["fixed_mask"].astype(np.float32)

    solid = density_np > FEMDatasetV3.SOLID_THRESHOLD
    if solid.any() and (~solid).any():
        from scipy.ndimage import distance_transform_edt

        sdf_np = (
            distance_transform_edt(solid) - distance_transform_edt(~solid)
        ).astype(np.float32) / (density_np.shape[0] / 2.0)
    else:
        sdf_np = np.zeros_like(density_np, dtype=np.float32)

    density = torch.from_numpy(density_np)[None, None].to(device)
    force = torch.from_numpy(force_np / FEMDatasetV3.FORCE_SCALE)[None].to(device)
    sdf = torch.from_numpy(sdf_np)[None, None].to(device)
    fixed = torch.from_numpy(fixed_np)[None].to(device)
    fields, log_vm_max = model.forward_with_scale(
        torch.cat([density, force, sdf, fixed], dim=1)
    )
    vm_shape = compute_von_mises(fields[:, 3:].float())[0, 0]
    scale = 10.0 ** float(log_vm_max[0, 0].item())
    vm_oriented = vm_shape.detach().cpu().numpy().astype(np.float32) * scale
    vm_max = float(vm_oriented[np.asarray(oriented_volume, dtype=bool)].max())
    return {
        "sf_pino": float(PINO_YIELD_STRESS / max(vm_max, 1e-12)),
        "vm_max_pino": vm_max,
        "vm_oriented": vm_oriented,
    }


def delivered_state(
    model: torch.nn.Module,
    core: np.ndarray,
    hull_256: np.ndarray,
    shell_256: np.ndarray,
    device: torch.device,
) -> dict:
    """Apply the production delivery/orientation/proxy path before PINO."""
    delivery = _generic_connected_delivery(
        core, hull_256, shell_256, include_shell=True
    )
    oriented = dict(print_orientations(delivery))["display_y_down"]
    oriented_hull = dict(print_orientations(hull_256))["display_y_down"]
    proxy, _ = structural_proxy(oriented, oriented_hull, 48)
    state = pino_state_oriented(model, proxy, device)
    state["vm_display"] = undo_orient_y_down(state.pop("vm_oriented"))
    state["delivery"] = delivery
    state["proxy"] = proxy
    return state


def projection_retention(volume: np.ndarray, reference: np.ndarray) -> list[float]:
    result = []
    for axis in range(3):
        target = reference.any(axis=axis)
        pred = volume.any(axis=axis)
        result.append(float(np.logical_and(pred, target).sum() / max(int(target.sum()), 1)))
    return result


def boundary(volume: np.ndarray) -> np.ndarray:
    occ = np.asarray(volume, dtype=bool)
    pad = np.pad(occ, 1)
    interior = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1]
        & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    return np.argwhere(occ & ~interior)


def tolerant_batch(
    volume: np.ndarray,
    reference: np.ndarray,
    score: np.ndarray,
    min_retention: float,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Remove one connectivity-safe batch under a global projection budget."""
    occ = np.asarray(volume, dtype=bool).copy()
    covers = [occ.sum(axis=axis) for axis in range(3)]
    target_counts = [max(int(reference.any(axis=axis).sum()), 1) for axis in range(3)]
    covered_counts = [int((cover > 0).sum()) for cover in covers]
    points = boundary(occ)
    if points.size:
        values = score[points[:, 0], points[:, 1], points[:, 2]]
        points = points[np.argsort(values, kind="stable")]

    removed = 0
    for x, y, z in points:
        if removed >= batch_size or not occ[x, y, z]:
            break
        indices = ((y, z), (x, z), (x, y))
        allowed = True
        for axis, index in enumerate(indices):
            if covers[axis][index] == 1:
                next_ratio = (covered_counts[axis] - 1) / target_counts[axis]
                if next_ratio < min_retention:
                    allowed = False
                    break
        if not allowed or not local_simple_point(occ, int(x), int(y), int(z)):
            continue
        occ[x, y, z] = False
        for axis, index in enumerate(indices):
            if covers[axis][index] == 1:
                covered_counts[axis] -= 1
            covers[axis][index] -= 1
        removed += 1
    return occ, removed


def run_variant(
    name: str,
    start: np.ndarray,
    hull: np.ndarray,
    hull_256: np.ndarray,
    shell_256: np.ndarray,
    targets_256: list[np.ndarray],
    metric_args: Namespace,
    guidance: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    min_retention: float,
    sf_threshold: float,
    sf_margin: float,
    batch_size: int,
    stress_weight: float,
    max_batches: int,
    out: Path,
) -> dict:
    current = np.asarray(start, dtype=bool).copy()
    guide_norm = normalize(guidance, hull)
    rows = []
    state = delivered_state(model, current, hull_256, shell_256, device)
    rows.append({
        "batch": 0,
        "occupied_voxels": int(current.sum()),
        "saving_vs_hull": float(1.0 - current.sum() / max(int(hull.sum()), 1)),
        "view_retention_vs_start": projection_retention(current, start),
        "sf_pino": state["sf_pino"],
        "accepted": True,
    })

    for batch in range(1, max_batches + 1):
        stress_norm = normalize(state["vm_display"], current)
        score = (1.0 - stress_weight) * guide_norm + stress_weight * stress_norm
        proposal, removed = tolerant_batch(
            current, start, score, min_retention=min_retention, batch_size=batch_size
        )
        if removed == 0:
            rows.append({"batch": batch, "removed": 0, "accepted": False, "reason": "geometry_fixed_point"})
            break
        if label(proposal, structure=S6)[1] != 1:
            rows.append({"batch": batch, "removed": removed, "accepted": False, "reason": "connectivity"})
            break
        proposal_state = delivered_state(
            model, proposal, hull_256, shell_256, device
        )
        accepted = proposal_state["sf_pino"] >= sf_threshold + sf_margin
        rows.append({
            "batch": batch,
            "removed": int(removed),
            "occupied_voxels": int(proposal.sum()),
            "saving_vs_hull": float(1.0 - proposal.sum() / max(int(hull.sum()), 1)),
            "view_retention_vs_start": projection_retention(proposal, start),
            "sf_pino": proposal_state["sf_pino"],
            "accepted": bool(accepted),
            "reason": "accepted" if accepted else "pino_stop_and_rollback",
        })
        if not accepted:
            break
        current = proposal
        state = proposal_state

    np.save(out / f"{name}_final_core48.npy", current.astype(np.uint8))
    fem = fem_sf(state["proxy"], load_multiplier=1.0)
    metrics = _generic_metrics(
        state["delivery"], targets_256, int(hull_256.sum()), metric_args
    )
    final = rows[-1] if rows[-1].get("accepted") else rows[-2]
    return {
        "variant": name,
        "stress_weight": stress_weight,
        "min_projection_retention": min_retention,
        "sf_stop": sf_threshold + sf_margin,
        "start_voxels": int(start.sum()),
        "final_voxels": int(current.sum()),
        "extra_removed": int(start.sum() - current.sum()),
        "core_saving_vs_hull48": float(1.0 - current.sum() / max(int(hull.sum()), 1)),
        "delivery_saving_vs_hull256": metrics["saving"],
        "delivery_mean_iou": metrics["mean_iou"],
        "delivery_min_iou": metrics["min_iou"],
        "delivery_mean_dice": metrics["mean_dice"],
        "view_retention_vs_start": projection_retention(current, start),
        "components": int(label(current, structure=S6)[1]),
        "sf_pino": float(state["sf_pino"]),
        "sf_fem": fem["sf_fem"],
        "fem_safe": fem["safe"],
        "stop_reason": rows[-1].get("reason", final.get("reason")),
        "trajectory": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--silhouettes", nargs=3, type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=160)
    ap.add_argument("--sf-threshold", type=float, default=1.3)
    ap.add_argument("--sf-margin", type=float, default=0.10)
    ap.add_argument("--retentions", default="0.90,0.80")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_pino(str(args.checkpoint), device)
    probability = np.load(args.artifact / "occupancy_probability_core.npy")
    probability_core = downsample_max_to(probability, 48)
    generic = Namespace(
        silhouettes=args.silhouettes,
        tsf=1,
        mr=1,
        device=args.device,
        print_mm=210.0,
    )
    targets = _generic_targets(args.artifact, 48)
    targets_256 = _generic_targets(args.artifact, 256)
    hull_zyx = _generic_visual_hull(targets, generic, 48)
    hull_256_zyx = _generic_visual_hull(targets_256, generic, 256)
    hull = np.transpose(hull_zyx, (2, 1, 0))
    hull_256 = np.transpose(hull_256_zyx, (2, 1, 0))
    shell_256 = hull_256 & ~binary_erosion(hull_256, structure=S6, iterations=2)
    guidance = np.transpose(probability_core, (2, 1, 0))
    seeds = _generic_seed_trajectory(
        guidance, hull, (args.threshold,), bridge_radius=2, bridge_passes=48
    )
    seed_name, seed = seeds[0]
    strict = _generic_tfs_carve_trajectory(
        seed, guidance, hull, seed_name, num_axes=3, bridge_radius=2, bridge_passes=48
    )
    strict_name, strict_fixed_point = strict[-1]
    np.save(args.output / "strict_fixed_point_core48.npy", strict_fixed_point.astype(np.uint8))

    strict_pino_score = delivered_state(
        model, strict_fixed_point, hull_256, shell_256, device
    )
    strict_soft_once, strict_soft_removed = axis_guarded_carve_step(
        strict_fixed_point, priority=guidance, num_axes=3
    )
    strict_stress_priority = (
        0.55 * normalize(guidance, hull)
        + 0.45 * normalize(strict_pino_score["vm_display"], strict_fixed_point)
    )
    strict_pino_once, strict_pino_removed = axis_guarded_carve_step(
        strict_fixed_point, priority=strict_stress_priority, num_axes=3
    )

    variants = []
    for retention in [float(x) for x in args.retentions.split(",")]:
        variants.append(run_variant(
            f"soft_only_r{retention:.2f}",
            strict_fixed_point, hull, hull_256, shell_256, targets_256, generic,
            guidance, model, device, retention,
            args.sf_threshold, args.sf_margin, args.batch_size, 0.0,
            args.max_batches, args.output,
        ))
        variants.append(run_variant(
            f"pino_guided_r{retention:.2f}",
            strict_fixed_point, hull, hull_256, shell_256, targets_256, generic,
            guidance, model, device, retention,
            args.sf_threshold, args.sf_margin, args.batch_size, 0.45,
            args.max_batches, args.output,
        ))

    payload = {
        "schema": "pcraft_pino_closed_loop_pilot_v1",
        "case": "connected_06",
        "protocol": {
            "soft_artifact": str(args.artifact),
            "checkpoint": str(args.checkpoint),
            "resolution": 48,
            "views": 3,
            "mr": 1,
            "seed_threshold": args.threshold,
            "sf_threshold": args.sf_threshold,
            "sf_margin": args.sf_margin,
            "batch_size": args.batch_size,
            "print_orientation": "display_y_down",
        },
        "strict_fixed_point": {
            "candidate": strict_name,
            "hull_voxels": int(hull.sum()),
            "occupied_voxels": int(strict_fixed_point.sum()),
            "saving_vs_hull": float(1.0 - strict_fixed_point.sum() / max(int(hull.sum()), 1)),
            "view_retention_vs_seed": projection_retention(strict_fixed_point, seed),
            "sf_pino": strict_pino_score["sf_pino"],
            "soft_priority_extra_removed": int(strict_soft_removed),
            "pino_priority_extra_removed": int(strict_pino_removed),
        },
        "variants": variants,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
