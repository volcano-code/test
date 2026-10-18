#!/usr/bin/env python3
"""Unified ShadowForge entry point for declared multi-view protocols.

The canonical soft optimizer accepts arbitrary view counts under the generic
orthographic protocol and the source TFS axis preset for one to three base
views. Physical delivery uses one common soft-seed, declared-ray restoration,
bridge repair, guarded carving, shell, mesh-QA, and structural-screen contract.
The unmirrored two-wall case additionally retains its wall-corner-specific
repair operator; every other declared protocol uses the generic backprojection
adapter instead of falling back to proposal-only output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label, zoom

ROOT = Path(__file__).resolve().parents[1]
for module_path in (ROOT, ROOT / "generation", ROOT / "experiments"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
from voxel_ops import sdf_watertight_mesh
from shadowforge_pipeline.run import (
    StructuralScreen,
    apply_pino_ranked_fem,
    apply_pino_rollback_fem,
)
BASE = ROOT / "experiments/build_anime_two_wall_upright_saving_print_10cm_256_20260710.py"
VOXEL_ROOT = Path("/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel")
if str(VOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(VOXEL_ROOT))
from camera_voxel import build_views
from renderer_voxel import render_orthographic
spec = importlib.util.spec_from_file_location("shadowforge_two_wall", BASE)
tw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tw)
n2 = tw.n2
S6 = generate_binary_structure(3, 1)
S2 = np.ones((3, 3), dtype=bool)


def _foreground_bbox_aspect(path: Path) -> float:
    """Return foreground width/height while tolerating either mask polarity."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    foreground = binary > 0
    if foreground.mean() > 0.5:
        foreground = ~foreground
    yy, xx = np.where(foreground)
    if not len(xx):
        raise ValueError(f"Target silhouette is empty: {path}")
    return float((xx.max() - xx.min() + 1) / (yy.max() - yy.min() + 1))


def _audit_preserved_target_aspects(
    source_paths: list[Path],
    train_dir: Path,
    tolerance: float = 0.03,
) -> list[dict[str, float | str]]:
    """Reject a soft artifact that silently changes a declared silhouette ratio."""
    rows: list[dict[str, float | str]] = []
    for index, source_path in enumerate(source_paths):
        normalized_path = train_dir / f"gt_view_{index}.png"
        source_ratio = _foreground_bbox_aspect(source_path)
        normalized_ratio = _foreground_bbox_aspect(normalized_path)
        relative_error = abs(normalized_ratio / max(source_ratio, 1e-12) - 1.0)
        rows.append({
            "view": int(index),
            "source": str(source_path),
            "normalized": str(normalized_path),
            "source_bbox_width_over_height": source_ratio,
            "normalized_bbox_width_over_height": normalized_ratio,
            "relative_error": relative_error,
        })
        if relative_error > tolerance:
            raise RuntimeError(
                "Target aspect-ratio drift after soft preprocessing for view "
                f"{index}: source={source_ratio:.6f}, "
                f"normalized={normalized_ratio:.6f}, "
                f"relative_error={relative_error:.3%}"
            )
    return rows


def support_metrics(volume: np.ndarray, print_mm: float) -> dict:
    """Same conservative 45-degree previous-layer proxy for every n=2 branch."""
    v = np.asarray(volume, dtype=bool)
    coords = np.where(v)
    if not coords[0].size:
        return {"unsupported_ratio": 1.0, "significant_islands": 0, "bed_area_mm2": 0.0}
    z0, z1 = int(coords[2].min()), int(coords[2].max())
    unsupported_total = islands = 0
    for z in range(z0 + 1, z1 + 1):
        unsupported = v[:, :, z] & ~binary_dilation(v[:, :, z - 1], structure=S2)
        unsupported_total += int(unsupported.sum())
        labs, count = label(unsupported, structure=S2)
        for idx in range(1, count + 1):
            pts = np.argwhere(labs == idx)
            if pts.size and (len(pts) >= 16 or (np.ptp(pts, axis=0).max() + 1) * print_mm / v.shape[0] >= 4.0):
                islands += 1
    return {
        "unsupported_ratio": float(unsupported_total / max(int(v.sum()), 1)),
        "significant_islands": int(islands),
        "bed_area_mm2": float(v[:, :, z0].sum() * (print_mm / v.shape[0]) ** 2),
    }


def print_orientations(volume: np.ndarray):
    for name, oriented in (
        ("display_z_up", volume),
        ("display_z_down", volume[:, :, ::-1]),
        ("display_x_up", np.transpose(volume, (1, 2, 0))),
        ("display_x_down", np.transpose(volume, (1, 2, 0))[:, :, ::-1]),
        ("display_y_up", np.transpose(volume, (0, 2, 1))),
        ("display_y_down", np.transpose(volume, (0, 2, 1))[:, :, ::-1]),
    ):
        yield name, oriented.copy()


def mesh_qa(volume: np.ndarray) -> tuple[bool, int, int]:
    """Report all SDF shells and the number of exterior shells separately."""
    try:
        mesh = sdf_watertight_mesh(np.transpose(volume, (2, 1, 0)), smooth_sigma=0.6)
        parts = mesh.split(only_watertight=False)
        if not parts:
            return bool(mesh.is_watertight), 0, 0
        global_min, global_max = mesh.bounds
        outer_parts = 0
        for part in parts:
            pmin, pmax = part.bounds
            outer_parts += int(bool(np.any(np.isclose(pmin, global_min, atol=1e-3))
                                    or np.any(np.isclose(pmax, global_max, atol=1e-3))))
        return bool(mesh.is_watertight and mesh.is_winding_consistent), int(len(parts)), int(outer_parts)
    except Exception:
        return False, 0, 0


def canonical_voxel_soft(args: argparse.Namespace, output: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the mesh8 voxel proposal trainer and return physical [X,Y,Z] fields.

    stdout is intentionally inherited so that every ``--save-every`` record is
    visible to the caller; this replaces the silent temporary optimizer.
    """
    if args.soft_artifact_dir is not None:
        train_dir = args.soft_artifact_dir.resolve()
        metrics_path = train_dir / "metrics.json"
        probability_path = train_dir / "occupancy_probability_core.npy"
        proposal_path = train_dir / "occupancy_soft_proposal.npy"
        expected_views = len(args.silhouettes) * (2 if args.mr else 1)
        required = (
            metrics_path,
            probability_path,
            proposal_path,
            *(train_dir / f"gt_view_{index}.png" for index in range(expected_views)),
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Frozen soft artifact is incomplete: " + ", ".join(missing)
            )
        probability_zyx = np.load(probability_path).astype(np.float32)
        proposal_zyx = np.load(proposal_path).astype(bool)
        expected = (args.soft_resolution,) * 3
        if probability_zyx.shape != expected or proposal_zyx.shape != expected:
            raise ValueError(
                f"Frozen soft output must be {expected}, got "
                f"{probability_zyx.shape}/{proposal_zyx.shape}"
            )
        # The trainer's image convention reverses both projected image axes
        # relative to the fixed two-wall geometry frame.  The permutation alone
        # leaves both silhouettes mirrored; apply the same global frame mapping
        # to every task (this is not per-case registration).
        probability_xyz = np.flip(np.transpose(probability_zyx, (2, 1, 0)), axis=(0, 2))
        proposal_xyz = np.flip(np.transpose(proposal_zyx, (2, 1, 0)), axis=(0, 2))
        aspect_audit = (
            _audit_preserved_target_aspects(args.silhouettes, train_dir)
            if args.soft_preserve_target_aspect else None
        )
        return (
            probability_xyz,
            proposal_xyz,
            {
                "backend": "frozen_mesh8_voxel_artifact",
                "command": None,
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "artifact_dir": str(train_dir),
                "preserve_target_aspect": bool(args.soft_preserve_target_aspect),
                "target_aspect_audit": aspect_audit,
            },
        )
    if not VOXEL_ROOT.is_dir():
        raise FileNotFoundError(f"Canonical voxel backend is missing: {VOXEL_ROOT}")
    # The canonical trainer runs with VOXEL_ROOT as cwd. Pass an absolute
    # artifact directory so a relative unified --output never writes outside
    # ShadowForge and then appears missing to this wrapper.
    train_dir = output.resolve() / "soft_training"
    command = [
        str(args.voxel_python), "train.py", args.device, str(train_dir),
        str(args.soft_iterations), str(args.soft_lr), "1", "-sdlist",
        *[str(path.resolve()) for path in args.silhouettes],
        "-mr", str(args.mr), "-tsf", str(args.tsf), "-tsf4", "0",
        "-cam", "ortho", "-imsz", "256", "--res", str(args.soft_resolution),
        "-eres", "256", "--save-every", str(args.soft_save_every),
        "--control-res", str(args.soft_control_res),
        "--proposal-mode", args.soft_proposal_mode,
        "--connectivity-start-iou", str(args.soft_connectivity_start_iou),
        "--fidelity-gate", str(args.soft_fidelity_gate),
        "--structure-warmup-steps", str(args.soft_structure_warmup_steps),
        "--structure-ramp-steps", str(args.soft_structure_ramp_steps),
        "--seed", str(args.soft_seed),
    ]
    disabled_losses = [item.strip() for item in args.soft_disable_losses.split(",") if item.strip()]
    if disabled_losses:
        command.extend(["--disable-losses", *disabled_losses])
    if args.soft_export_final_step:
        # Ablations must compare a common optimization endpoint. Selecting by
        # thresholded training projection IoU can prefer a fragmented field as
        # soon as a few voxels cross the binary threshold.
        command.append("--export-final-step")
    if args.soft_preserve_target_aspect:
        command.append("--preserve-target-aspect")
    subprocess.run(command, cwd=VOXEL_ROOT, check=True)
    metrics_path = train_dir / "metrics.json"
    probability_path = train_dir / "occupancy_probability_core.npy"
    proposal_path = train_dir / "occupancy_soft_proposal.npy"
    if not (metrics_path.is_file() and probability_path.is_file() and proposal_path.is_file()):
        raise RuntimeError("Canonical voxel trainer completed without its required proposal artifacts")
    probability_zyx = np.load(probability_path).astype(np.float32)
    proposal_zyx = np.load(proposal_path).astype(bool)
    expected = (args.soft_resolution,) * 3
    if probability_zyx.shape != expected or proposal_zyx.shape != expected:
        raise ValueError(f"Canonical soft output must be {expected}, got {probability_zyx.shape}/{proposal_zyx.shape}")
    # The canonical trainer stores [Z,Y,X] and reverses both projected image
    # axes relative to the fixed two-wall geometry convention.
    probability_xyz = np.flip(np.transpose(probability_zyx, (2, 1, 0)), axis=(0, 2))
    proposal_xyz = np.flip(np.transpose(proposal_zyx, (2, 1, 0)), axis=(0, 2))
    aspect_audit = (
        _audit_preserved_target_aspects(args.silhouettes, train_dir)
        if args.soft_preserve_target_aspect else None
    )
    return (probability_xyz,
            proposal_xyz,
            {"backend": "mesh8_voxel", "command": command,
             "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
             "artifact_dir": str(train_dir),
             "preserve_target_aspect": bool(args.soft_preserve_target_aspect),
             "target_aspect_audit": aspect_audit})


def soft_volume(left: np.ndarray, right: np.ndarray, resolution: int, iterations: int, lr: float, device: str) -> tuple[np.ndarray, dict]:
    """Differentiable fixed-camera occupancy proposal; no hull is used here."""
    dev = torch.device(device)
    left_t = torch.as_tensor(left.astype(np.float32), device=dev)
    right_t = torch.as_tensor(right.astype(np.float32), device=dev)
    logits = torch.full((resolution, resolution, resolution), -4.0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    history = []
    for step in range(1, iterations + 1):
        prob = torch.sigmoid(logits)
        pred_left = prob.amax(dim=0)
        pred_right = prob.amax(dim=1)
        bce = torch.nn.functional.binary_cross_entropy(pred_left, left_t) + torch.nn.functional.binary_cross_entropy(pred_right, right_t)
        dice = 1.0 - (2 * (pred_left * left_t).sum() + 1) / (pred_left.sum() + left_t.sum() + 1)
        dice += 1.0 - (2 * (pred_right * right_t).sum() + 1) / (pred_right.sum() + right_t.sum() + 1)
        tv = (prob[1:] - prob[:-1]).abs().mean() + (prob[:, 1:] - prob[:, :-1]).abs().mean() + (prob[:, :, 1:] - prob[:, :, :-1]).abs().mean()
        loss = bce + dice + 0.003 * tv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, iterations // 10) == 0 or step == iterations:
            history.append({"step": step, "loss": float(loss.detach()), "bce": float(bce.detach()), "dice_loss": float(dice.detach())})
    probability = torch.sigmoid(logits).detach().cpu().numpy()
    # The threshold is selected after geometry-compatible projection repair,
    # rather than by a free camera score.
    return probability, {"iterations": iterations, "lr": lr, "history": history}


def upsample_nearest_to(volume: np.ndarray, target_resolution: int = 256) -> np.ndarray:
    """Nearest-neighbor lifting for arbitrary core/delivery resolution pairs.

    In particular, this implements the required 48^3 -> 256^3 mapping without
    assuming that the delivery size is an integer multiple of the core size.
    """
    source = np.asarray(volume)
    if source.shape == (target_resolution,) * 3:
        return source.copy()
    idx = np.floor(np.arange(target_resolution, dtype=np.float32) * source.shape[0] / target_resolution).astype(np.int64)
    idx = np.clip(idx, 0, source.shape[0] - 1)
    return source[np.ix_(idx, idx, idx)]


def downsample_max_to(volume: np.ndarray, target_resolution: int) -> np.ndarray:
    """Conservatively transfer a dense proposal to the structural core grid."""
    # Camera-frame normalization may flip an axis and therefore produce a
    # negative-stride view.  PyTorch cannot wrap such arrays directly.
    source = np.ascontiguousarray(volume, dtype=np.float32)
    if source.shape == (target_resolution,) * 3:
        return source.copy()
    tensor = torch.from_numpy(source)[None, None]
    return F.adaptive_max_pool3d(tensor, output_size=(target_resolution,) * 3)[0, 0].numpy()


def repair_two_wall(
    volume: np.ndarray,
    hull: np.ndarray,
    guidance: np.ndarray | None = None,
) -> np.ndarray:
    """Restore missing two-wall ray support inside the hull.

    Without guidance this preserves the historical deterministic midpoint rule.
    A soft-origin candidate instead restores each ray at its highest-density
    admissible voxel, making the repair faithful to the learned proposal.
    """
    if guidance is None:
        return tw.repair_projection(volume & hull, hull)
    occ = tw.tw.keep_largest(np.asarray(volume, dtype=bool) & hull)
    score = np.asarray(guidance, dtype=np.float32)
    if score.shape != hull.shape:
        raise ValueError(f"repair guidance shape {score.shape} does not match hull {hull.shape}")
    for y, z in np.argwhere(tw.tw.project_left_wall_x(hull) & ~tw.tw.project_left_wall_x(occ)):
        xs = np.flatnonzero(hull[:, y, z])
        if xs.size:
            occ[int(xs[np.argmax(score[xs, y, z])]), y, z] = True
    for x, z in np.argwhere(tw.tw.project_right_wall_y(hull) & ~tw.tw.project_right_wall_y(occ)):
        ys = np.flatnonzero(hull[x, :, z])
        if ys.size:
            occ[x, int(ys[np.argmax(score[x, ys, z])]), z] = True
    return tw.tw.keep_largest(occ & hull)


def local_simple_point(volume: np.ndarray, x: int, y: int, z: int) -> bool:
    """Whether deleting a voxel preserves local face connectivity."""
    x0, y0, z0 = max(x - 1, 0), max(y - 1, 0), max(z - 1, 0)
    block = volume[x0:x + 2, y0:y + 2, z0:z + 2].copy()
    cx, cy, cz = x - x0, y - y0, z - z0
    block[cx, cy, cz] = False
    labs, _ = label(block, structure=S6)
    neighbor_labels = set()
    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
        px, py, pz = cx + dx, cy + dy, cz + dz
        if 0 <= px < block.shape[0] and 0 <= py < block.shape[1] and 0 <= pz < block.shape[2] and block[px, py, pz]:
            neighbor_labels.add(int(labs[px, py, pz]))
    return len(neighbor_labels) <= 1


def two_wall_carve_step(volume: np.ndarray, priority: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Delete only projection-redundant, locally simple voxels for two walls.

    Unlike the three-view carve, only the X and Y projections are preserved:
    the floor/top projection is deliberately absent in the two-wall task.
    """
    occ = np.asarray(volume, dtype=bool).copy()
    cover_left = occ.sum(axis=0)   # [Y, Z]
    cover_right = occ.sum(axis=1)  # [X, Z]
    removed = 0
    # Boundary voxels are tried first; this preserves a thick soft proposal
    # while removing only rays that have redundant occupancy.
    pad = np.pad(occ, 1)
    interior = (pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] &
                pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] &
                pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2])
    boundary = np.argwhere(occ & ~interior)
    if priority is not None and len(boundary):
        # Low soft density means low learned importance and is considered
        # first. The hard ray/connectivity guards below remain authoritative.
        scores = priority[boundary[:, 0], boundary[:, 1], boundary[:, 2]]
        boundary = boundary[np.argsort(scores, kind="stable")]
    for x, y, z in boundary:
        if not occ[x, y, z] or cover_left[y, z] <= 1 or cover_right[x, z] <= 1:
            continue
        if not local_simple_point(occ, int(x), int(y), int(z)):
            continue
        occ[x, y, z] = False
        cover_left[y, z] -= 1
        cover_right[x, z] -= 1
        removed += 1
    return occ, removed


def axis_guarded_carve_step(
    volume: np.ndarray,
    priority: np.ndarray | None = None,
    num_axes: int = 3,
    max_removals: int | None = None,
) -> tuple[np.ndarray, int]:
    """Delete redundant voxels while preserving every declared TFS axis ray.

    ``max_removals`` turns one boundary sweep into a small material checkpoint.
    It is used by gate-driven trajectories so fabrication and structural gates
    can be evaluated before a large batch of admissible deletions is committed.
    """
    if num_axes not in (1, 2, 3):
        raise ValueError(f"Strict TFS carving supports one to three axes, got {num_axes}")
    occ = np.asarray(volume, dtype=bool).copy()
    covers = [occ.sum(axis=axis) for axis in range(num_axes)]
    pad = np.pad(occ, 1)
    interior = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1]
        & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    boundary = np.argwhere(occ & ~interior)
    if priority is not None and len(boundary):
        scores = priority[boundary[:, 0], boundary[:, 1], boundary[:, 2]]
        boundary = boundary[np.argsort(scores, kind="stable")]

    removed = 0
    for x, y, z in boundary:
        if not occ[x, y, z]:
            continue
        ray_indices = ((y, z), (x, z), (x, y))
        if any(covers[axis][ray_indices[axis]] <= 1 for axis in range(num_axes)):
            continue
        if not local_simple_point(occ, int(x), int(y), int(z)):
            continue
        occ[x, y, z] = False
        for axis in range(num_axes):
            covers[axis][ray_indices[axis]] -= 1
        removed += 1
        if max_removals is not None and removed >= max_removals:
            break
    return occ, removed


def connect_two_wall(
    volume: np.ndarray,
    hull: np.ndarray,
    max_passes: int = 8,
    bridge_radius: int = 1,
    restore_rays: bool = True,
    guidance: np.ndarray | None = None,
) -> np.ndarray:
    """Use the shared 6-connected bridge primitive, then restore wall rays."""
    from voxel_ops import connect6_min_bridges
    connected, _ = connect6_min_bridges(
        volume, hull, max_passes=max_passes, bridge_radius=bridge_radius
    )
    return repair_two_wall(connected, hull, guidance=guidance) if restore_rays else connected


def soft_guided_hull_trajectory(guidance: np.ndarray, hull: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Full-hull scaffold followed by soft-prioritized guarded reduction.

    The learned field never removes material before hard geometry begins. It
    only orders otherwise admissible deletions from the complete H48 scaffold.
    """
    scaffold = connect_two_wall(repair_two_wall(hull, hull), hull)
    print(json.dumps({"stage": "geometry_scaffold", "candidate": "soft_guided:full_hull_scaffold",
                      "occupied_voxels": int(scaffold.sum())}), flush=True)
    trajectory = [("soft_guided:full_hull_scaffold", scaffold.copy())]
    current = scaffold
    for phase, rounds in enumerate((1, 2, 4, 8, 16, 32), start=1):
        removed_total = 0
        for _ in range(rounds):
            current, removed = two_wall_carve_step(current, priority=guidance)
            removed_total += removed
            if not removed:
                break
        current = connect_two_wall(current, hull)
        name = f"soft_guided:carve_phase_{phase}_removed_{removed_total}"
        print(json.dumps({"stage": "geometry_carve", "phase": phase,
                          "candidate": name, "removed_voxels": int(removed_total),
                          "occupied_voxels": int(current.sum())}), flush=True)
        trajectory.append((name, current.copy()))
        if not removed_total:
            break
    return trajectory


def soft_seeded_scaffold(
    guidance: np.ndarray,
    hull: np.ndarray,
    threshold: float,
    restore_rays: bool = True,
    bridge: bool = True,
    bridge_radius: int = 1,
    bridge_passes: int = 8,
) -> tuple[str, np.ndarray]:
    """Build one connected soft-origin scaffold inside the hard envelope.

    The thresholded proposal is never exported directly.  It is clipped to the
    hard feasible envelope, receives missing ray support through repair_two_wall,
    and is bridged before delivery or guarded deletion is attempted.
    """
    seed = (np.asarray(guidance, dtype=np.float32) >= threshold) & hull
    scaffold = repair_two_wall(seed, hull, guidance=guidance) if restore_rays else seed
    if bridge:
        scaffold = connect_two_wall(
            scaffold, hull, max_passes=bridge_passes, bridge_radius=bridge_radius,
            restore_rays=restore_rays, guidance=guidance,
        )
    label = f"soft_seed_t{threshold:.2f}"
    print(json.dumps({"stage": "soft_seed_repair", "candidate": label,
                      "threshold": threshold, "seed_voxels": int(seed.sum()),
                      "scaffold_voxels": int(scaffold.sum())}), flush=True)
    return label, scaffold


def soft_seeded_geometry_trajectory(
    guidance: np.ndarray,
    hull: np.ndarray,
    threshold: float,
    restore_rays: bool = True,
    bridge: bool = True,
    bridge_radius: int = 1,
    bridge_passes: int = 8,
) -> list[tuple[str, np.ndarray]]:
    """Extend a repaired soft-origin scaffold with guarded material reduction."""
    label, scaffold = soft_seeded_scaffold(
        guidance, hull, threshold, restore_rays, bridge, bridge_radius, bridge_passes
    )
    trajectory = [(f"{label}:repaired", scaffold.copy())]
    current = scaffold
    for phase, rounds in enumerate((1, 2, 4, 8, 16, 32), start=1):
        removed_total = 0
        for _ in range(rounds):
            current, removed = two_wall_carve_step(current, priority=guidance)
            removed_total += removed
            if not removed:
                break
        if bridge:
            current = connect_two_wall(
                current, hull, max_passes=bridge_passes, bridge_radius=bridge_radius,
                restore_rays=restore_rays, guidance=guidance,
            )
        elif restore_rays:
            current = repair_two_wall(current, hull, guidance=guidance)
        name = f"{label}:carve_phase_{phase}_removed_{removed_total}"
        print(json.dumps({"stage": "geometry_carve", "phase": phase,
                          "candidate": name, "removed_voxels": int(removed_total),
                          "occupied_voxels": int(current.sum())}), flush=True)
        trajectory.append((name, current.copy()))
        if not removed_total:
            break
    return trajectory


def parse_soft_seed_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not thresholds:
        raise ValueError("--soft-seed-thresholds must contain at least one threshold")
    if any(threshold <= 0.0 or threshold >= 1.0 for threshold in thresholds):
        raise ValueError("--soft-seed-thresholds values must lie strictly between 0 and 1")
    return thresholds


def trajectory_metadata(candidate: str) -> tuple[str, int]:
    """Recover a stable heavy-to-light trajectory index from candidate names."""
    name = str(candidate)
    if ":core" in name:
        name = name.split(":core", 1)[0]
    marker = ":carve_phase_"
    if marker in name:
        prefix, suffix = name.split(marker, 1)
        phase_text = suffix.split("_", 1)[0]
        return prefix, int(phase_text)
    for suffix in (":repaired_only", ":repaired"):
        if suffix in name:
            return name.split(suffix, 1)[0], 0
    if name.endswith("full_hull_scaffold"):
        return name, 0
    return "candidate_pool", 0


def geometry_candidates(left: np.ndarray, right: np.ndarray):
    hull = tw.tw.visual_hull_two_wall(left, right)
    yield "geometry:hull", tw.tw.keep_largest(hull)
    for offset in (0, 7, 17, 31, 47):
        seed = tw.sparse_cover(left, right, offset)
        for dilation in (10, 12, 14, 16, 18, 20, 22, 24, 28, 32):
            volume = binary_dilation(seed, n2.S6, iterations=dilation) & hull
            yield f"geometry:sparse_o{offset}_d{dilation}", repair_two_wall(volume, hull)


def evaluate(volume: np.ndarray, left: np.ndarray, right: np.ndarray, hull_voxels: int, print_mm: float) -> dict:
    volume = np.asarray(volume, dtype=bool)
    metrics = tw.tw.metrics_two_wall(volume, left, right, hull_voxels)
    watertight, mesh_parts, mesh_outer_parts = mesh_qa(volume)
    options = [(support_metrics(v, print_mm), name, v) for name, v in print_orientations(volume)]
    fdm, orientation, print_volume = min(options, key=lambda item: (item[0]["unsupported_ratio"], item[0]["significant_islands"], -item[0]["bed_area_mm2"]))
    return {
        **metrics,
        "watertight": watertight,
        "mesh_parts": mesh_parts,
        "mesh_outer_parts": mesh_outer_parts,
        "mesh_internal_parts": int(max(mesh_parts - mesh_outer_parts, 0)),
        "print_orientation": orientation,
        "fdm_proxy": fdm,
        "print_volume": print_volume,
    }


def _evidence_name(name: str) -> str:
    """Stable filesystem label for an evidence-stage name."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def save_two_wall_evidence(
    root: Path,
    name: str,
    volume: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    hull_voxels: int,
) -> Path:
    """Persist real voxel and two-view evidence for a pipeline stage.

    These files are deliberately exported from the same volumes used by the
    selector, rather than being redrawn as explanatory artwork.
    """
    stage = root / _evidence_name(name)
    stage.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(volume, dtype=bool)
    np.save(stage / "volume.npy", volume.astype(np.uint8))
    pred_left = tw.wall_axes_to_image(tw.tw.project_left_wall_x(volume))
    pred_right = tw.wall_axes_to_image(tw.tw.project_right_wall_y(volume))
    n2.save_mask(pred_left, stage / "projection_view1.png")
    n2.save_mask(pred_right, stage / "projection_view2.png")
    tw.tw.contact_sheet(name, left, right, pred_left, pred_right, stage / "target_vs_projection.png")
    metrics = tw.tw.metrics_two_wall(volume, left, right, hull_voxels)
    (stage / "stage.json").write_text(json.dumps({
        "stage": name,
        "resolution": int(volume.shape[0]),
        "occupied_voxels": int(volume.sum()),
        **{key: float(value) if isinstance(value, (np.floating, float)) else value
           for key, value in metrics.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stage


def fabrication_eligible(row: dict, args: argparse.Namespace) -> bool:
    """Hard visual and fabrication gate, independent of structural screening."""
    return bool(
        row["mean_iou"] >= args.min_iou
        and row["min_iou"] >= args.min_view_iou
        and fabrication_eligible_without_fidelity(row, args)
    )


def fabrication_eligible_without_fidelity(row: dict, args: argparse.Namespace) -> bool:
    """All delivery gates except mean and worst-view IoU.

    This is used only by the explicit best-effort fallback. It never relaxes
    material, topology, mesh, support, or structural requirements.
    """
    return bool(
        row["saving"] >= args.min_saving
        and row["saving"] <= args.max_saving
        and row["components"] == 1
        and row["erosion_survival"] >= args.min_erosion_survival
        and row["watertight"]
        # Enclosed cavity shells do not create a second physical body. The
        # exterior surface must nevertheless be unique and watertight.
        and row["mesh_outer_parts"] == 1
        and row["fdm_proxy"]["unsupported_ratio"] <= args.max_unsupported_ratio
        and row["fdm_proxy"]["significant_islands"] <= args.max_support_islands
        and row["fdm_proxy"]["bed_area_mm2"] >= args.min_bed_area_mm2
    )


def eligible(row: dict, args: argparse.Namespace) -> bool:
    """Final admission gate; structure is fail-closed only when requested."""
    if not fabrication_eligible(row, args):
        return False
    return args.screen == "skip" or bool(row.get("structural_pass", False))


def mark_selection_eligibility(rows: list[dict], args: argparse.Namespace) -> bool:
    """Mark strict or explicit IoU-only fallback admission.

    Returns True when the fallback was activated. Strict candidates always
    take precedence; fallback is considered only when the strict set is empty.
    """
    for row in rows:
        row["selection_eligible"] = bool(row.get("eligible", False))
        row["best_effort_fidelity"] = False
    if any(row["selection_eligible"] for row in rows):
        return False
    if not args.allow_best_effort_fidelity_fallback:
        return False
    for row in rows:
        structural_ok = args.screen == "skip" or bool(row.get("structural_pass", False))
        fallback_ok = bool(
            row.get("fabrication_pass_without_fidelity", False) and structural_ok
        )
        row["best_effort_fidelity"] = fallback_ok
        row["selection_eligible"] = fallback_ok
    return any(row["selection_eligible"] for row in rows)


def structural_proxy(
    print_volume: np.ndarray,
    print_hull: np.ndarray,
    target_resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative delivered proxy used by the established 48^3 screen.

    The max-pooling step retains every occupied delivery cell that overlaps a
    core cell.  Screening therefore refers to the printed orientation, rather
    than to an un-oriented display mesh or an unrelated pre-delivery core.
    """
    volume = downsample_max_to(np.asarray(print_volume, dtype=bool), target_resolution) >= 0.5
    hull = downsample_max_to(np.asarray(print_hull, dtype=bool), target_resolution) >= 0.5
    return volume, hull


def export_stl_at_print_scale(volume: np.ndarray, path: Path, print_mm: float):
    """Export the delivered volume through the production SDF surface path."""
    mesh = sdf_watertight_mesh(np.asarray(volume, dtype=bool), smooth_sigma=0.6)
    extent = float(np.max(mesh.extents))
    if extent <= 0:
        raise ValueError("Cannot scale an empty delivery mesh")
    mesh.apply_scale(float(print_mm) / extent)
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])
    mesh.export(path)
    return mesh


def select(rows: list[dict], args: argparse.Namespace) -> dict:
    valid = [row for row in rows if row.get("selection_eligible", row["eligible"])]
    pool = valid or rows
    # A balance point is selected from the material-reduction trajectory: retain
    # candidates within a small fidelity band.  Physical display delivery may
    # instead prefer a support-easier orientation; it does not claim that
    # temporary supports are absent.
    best_iou = max(r["mean_iou"] for r in pool)
    near_fidelity = [r for r in pool if r["mean_iou"] >= best_iou - args.selection_iou_tolerance]
    if args.selection_policy == "fdm":
        return min(
            near_fidelity,
            key=lambda r: (
                r["fdm_proxy"]["unsupported_ratio"],
                r["fdm_proxy"]["significant_islands"],
                -r["fdm_proxy"]["bed_area_mm2"],
                -r["saving"],
            ),
        )
    return max(
        near_fidelity,
        key=lambda r: (
            r.get("selection_eligible", r["eligible"]),
            r["saving"],
            r["min_iou"],
            -r["fdm_proxy"]["unsupported_ratio"],
        ),
    )


def run_two_wall(args: argparse.Namespace) -> None:
    left_img = n2.load_mask(args.silhouettes[0], 256)
    right_img = n2.load_mask(args.silhouettes[1], 256)
    left, right = tw.image_to_wall_axes(left_img), tw.image_to_wall_axes(right_img)
    hull = tw.tw.visual_hull_two_wall(left, right)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    n2.save_mask(left_img, out / "inputs/left_wall_x_target.png")
    n2.save_mask(right_img, out / "inputs/right_wall_y_target.png")
    candidates: list[tuple[str, np.ndarray]] = []
    soft_info = None
    # Both geometry-only and soft-guided branches can request delivery evidence.
    evidence_root = out / "candidate_evidence" if args.save_candidate_evidence else None
    if args.screen in {"pino_acs_fem", "pino_ranked_fem", "pino_rollback_fem"} and not args.pino_checkpoint.is_file():
        raise FileNotFoundError(f"PINO checkpoint not found: {args.pino_checkpoint}")
    structural_screen = StructuralScreen(args)
    if args.branch == "geometry":
        candidates.extend(geometry_candidates(left, right))
    if args.branch in ("soft", "soft-repair", "soft-geometry"):
        print(json.dumps({"stage": "soft_training", "resolution": args.soft_resolution,
                          "control_resolution": args.soft_control_res,
                          "proposal_mode": args.soft_proposal_mode}), flush=True)
        probability, canonical_proposal, soft_info = canonical_voxel_soft(args, out)
        canonical_dir = Path(soft_info["artifact_dir"])
        # Geometry must consume the exact normalized targets used by the soft
        # trainer; otherwise its visual hull is in a different image frame.
        left_img = n2.load_mask(canonical_dir / "gt_view_0.png", 256)
        right_img = n2.load_mask(canonical_dir / "gt_view_1.png", 256)
        left, right = tw.image_to_wall_axes(left_img), tw.image_to_wall_axes(right_img)
        hull = tw.tw.visual_hull_two_wall(left, right)
        n2.save_mask(left_img, out / "inputs/left_wall_x_target.png")
        n2.save_mask(right_img, out / "inputs/right_wall_y_target.png")
        core_left = tw.image_to_wall_axes(n2.load_mask(canonical_dir / "gt_view_0.png", args.core_resolution))
        core_right = tw.image_to_wall_axes(n2.load_mask(canonical_dir / "gt_view_1.png", args.core_resolution))
        probability_core = downsample_max_to(probability, args.core_resolution)
        np.save(out / "soft_probability_256.npy", probability.astype(np.float32))
        np.save(out / "soft_canonical_proposal_256.npy", canonical_proposal.astype(np.uint8))
        np.save(out / "soft_probability_core48.npy", probability_core.astype(np.float32))
        np.save(out / "soft_canonical_proposal_core48.npy",
                (probability_core >= args.soft_guidance_threshold).astype(np.uint8))
        if args.branch == "soft":
            # A soft run is an isolated 48^3 proposal experiment.  It is not
            # allowed to acquire 256^3 shell fidelity or an STL by accident.
            manifest = {
                "schema": "shadowforge_unified_v1", "status": "SOFT_PROPOSAL_ONLY",
                "view_model": "fixed_two_wall", "n_views": 2,
                "soft": soft_info,
                "reason": "No geometry repair, resolution-decoupled delivery, or STL export was requested.",
            }
            (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(out), "status": "SOFT_PROPOSAL_ONLY"}, ensure_ascii=False, indent=2))
            return
        def deliver(core_volume: np.ndarray) -> np.ndarray:
            core_up = upsample_nearest_to(core_volume, 256) & hull
            shell = (hull & ~binary_erosion(hull, structure=S6, iterations=args.delivery_shell_voxels)
                     if not args.disable_delivery_shell else np.zeros_like(hull, dtype=bool))
            # This is the authoritative resolution-decoupled delivery form:
            # (C48 upsampled and clipped to H256) union a 256 silhouette shell.
            delivered = core_up | shell
            if not args.disable_ray_restoration:
                delivered = repair_two_wall(delivered, hull, guidance=probability)
            if not args.disable_bridge:
                delivered = connect_two_wall(
                    delivered, hull, max_passes=args.bridge_passes, bridge_radius=args.bridge_radius,
                    restore_rays=not args.disable_ray_restoration,
                    guidance=probability,
                )
            return delivered
        core_hull = tw.tw.visual_hull_two_wall(core_left, core_right)
        if evidence_root is not None:
            save_two_wall_evidence(
                evidence_root, "core_hull_scaffold", core_hull,
                core_left, core_right, int(core_hull.sum()),
            )
        print(json.dumps({"stage": "soft_to_geometry", "core_resolution": args.core_resolution,
                          "core_hull_voxels": int(core_hull.sum())}), flush=True)
        thresholds = parse_soft_seed_thresholds(args.soft_seed_thresholds)
        if args.branch == "soft-repair":
            # Minimal deterministic export adapter: no full-hull trajectory and
            # no guarded material-reduction iterations are executed here.
            for threshold in thresholds:
                label_name, core_volume = soft_seeded_scaffold(
                    probability_core, core_hull, threshold,
                    restore_rays=not args.disable_ray_restoration,
                    bridge=not args.disable_bridge,
                    bridge_radius=args.bridge_radius,
                    bridge_passes=args.bridge_passes,
                )
                candidates.append((f"{label_name}:core{args.core_resolution}:delivery256", deliver(core_volume)))
        else:
            # The unified method is soft-seeded by construction.  The visual
            # hull remains a feasible envelope for clipping and repair, never
            # an independently admitted full-hull candidate trajectory.
            for threshold in thresholds:
                if evidence_root is not None:
                    raw_seed = (probability_core >= threshold) & core_hull
                    restored_seed = (repair_two_wall(raw_seed, core_hull, guidance=probability_core)
                                     if not args.disable_ray_restoration else raw_seed)
                    bridged_seed = (connect_two_wall(
                        restored_seed, core_hull,
                        max_passes=args.bridge_passes, bridge_radius=args.bridge_radius,
                        restore_rays=not args.disable_ray_restoration,
                        guidance=probability_core,
                    ) if not args.disable_bridge else restored_seed)
                    prefix = f"seed_t{threshold:.2f}"
                    save_two_wall_evidence(evidence_root, f"{prefix}_raw", raw_seed,
                                           core_left, core_right, int(core_hull.sum()))
                    save_two_wall_evidence(evidence_root, f"{prefix}_restore_rays", restored_seed,
                                           core_left, core_right, int(core_hull.sum()))
                    save_two_wall_evidence(evidence_root, f"{prefix}_bridge6", bridged_seed,
                                           core_left, core_right, int(core_hull.sum()))
                if args.disable_guarded_carve:
                    # Module-ablation control: retain the repaired soft scaffold,
                    # but do not run any guarded deletion round.
                    label_name, core_volume = soft_seeded_scaffold(
                        probability_core, core_hull, threshold,
                        restore_rays=not args.disable_ray_restoration,
                        bridge=not args.disable_bridge,
                        bridge_radius=args.bridge_radius,
                        bridge_passes=args.bridge_passes,
                    )
                    candidates.append((f"{label_name}:repaired_only:core{args.core_resolution}:delivery256",
                                       deliver(core_volume)))
                else:
                    for label_name, core_volume in soft_seeded_geometry_trajectory(
                        probability_core, core_hull, threshold,
                        restore_rays=not args.disable_ray_restoration,
                        bridge=not args.disable_bridge,
                        bridge_radius=args.bridge_radius,
                        bridge_passes=args.bridge_passes,
                    ):
                        candidates.append((f"{label_name}:core{args.core_resolution}:delivery256", deliver(core_volume)))
    rows = []
    for name, volume in candidates:
        data = evaluate(volume, left, right, int(hull.sum()), args.print_mm)
        data.update({"candidate": name, "eligible": False})
        data["fabrication_pass"] = fabrication_eligible(data, args)
        data["fabrication_pass_without_fidelity"] = fabrication_eligible_without_fidelity(data, args)
        data["structural_screen_admissible"] = bool(
            data["fabrication_pass"]
            or (
                args.allow_best_effort_fidelity_fallback
                and data["fabrication_pass_without_fidelity"]
            )
        )
        if args.screen == "skip":
            safety = {"status": "NOT_REQUESTED", "safe": None, "decided_by": "none"}
        elif not data["structural_screen_admissible"]:
            safety = {"status": "NOT_EVALUATED", "safe": False, "decided_by": "fabrication_gate"}
        else:
            orientations = dict(print_orientations(hull))
            print_hull = orientations[data["print_orientation"]]
            proxy, proxy_hull = structural_proxy(
                data["print_volume"], print_hull, args.structural_resolution
            )
            data["_structural_volume"] = proxy
            data["_structural_hull"] = proxy_hull
            data["_structural_material"] = int(data["print_volume"].sum())
            trajectory_id, trajectory_step = trajectory_metadata(name)
            data["_trajectory_id"] = trajectory_id
            data["_trajectory_step"] = trajectory_step
            if args.screen in {"pino_ranked_fem", "pino_rollback_fem"}:
                safety = {"status": "PENDING_PINO_POLICY", "safe": False,
                          "decided_by": "RAW_PINO_TRAJECTORY_POLICY"}
            else:
                try:
                    safety = structural_screen.evaluate(proxy, proxy_hull)
                except Exception as exc:
                    safety = {"status": "ERROR", "safe": False, "decided_by": "error", "error": repr(exc)}
            safety["structural_resolution"] = int(args.structural_resolution)
            safety["structural_representation"] = "max-pooled delivered print orientation"
        data["structural_screen"] = safety
        data["structural_pass"] = bool(safety.get("safe", False))
        data["safety_score"] = float(safety.get("safety_score", safety.get("sf_fem", 0.0)) or 0.0)
        data["eligible"] = eligible(data, args)
        rows.append(data)
        if args.screen not in {"pino_ranked_fem", "pino_rollback_fem"}:
            print(json.dumps({"stage": "delivery_qa", "candidate": name,
                          "mean_iou": data["mean_iou"], "saving": data["saving"],
                          "components": data["components"], "watertight": data["watertight"],
                          "mesh_outer_parts": data["mesh_outer_parts"],
                          "fabrication_pass": data["fabrication_pass"],
                          "structural_status": safety["status"],
                          "structural_decided_by": safety["decided_by"],
                          "eligible": data["eligible"]}), flush=True)
    ranked_fem_summary = None
    if args.screen == "pino_ranked_fem":
        ranked_fem_summary = apply_pino_ranked_fem(rows, structural_screen, args)
    elif args.screen == "pino_rollback_fem":
        ranked_fem_summary = apply_pino_rollback_fem(rows, structural_screen, args)
    if args.screen in {"pino_ranked_fem", "pino_rollback_fem"}:
        for data in rows:
            safety = data["structural_screen"]
            if data.get("fabrication_pass"):
                safety["structural_resolution"] = int(args.structural_resolution)
                safety["structural_representation"] = "max-pooled delivered print orientation"
            data["eligible"] = eligible(data, args)
            print(json.dumps({"stage": "delivery_qa", "candidate": data["candidate"],
                              "mean_iou": data["mean_iou"], "saving": data["saving"],
                              "components": data["components"], "watertight": data["watertight"],
                              "mesh_outer_parts": data["mesh_outer_parts"],
                              "fabrication_pass": data["fabrication_pass"],
                              "structural_status": safety["status"],
                              "structural_decided_by": safety["decided_by"],
                              "pino_rank": data.get("pino_rank"),
                              "pino_trajectory_rank": data.get("pino_trajectory_rank"),
                              "pino_monotone_sf": data.get("pino_monotone_sf"),
                              "fem_checked": data.get("fem_checked", False),
                              "eligible": data["eligible"]}), flush=True)
    fallback_activated = mark_selection_eligibility(rows, args)
    if evidence_root is not None:
        evidence_rows = []
        for name, volume in candidates:
            stage = save_two_wall_evidence(evidence_root / "delivery_candidates", name, volume,
                                           left, right, int(hull.sum()))
            evidence_rows.append({"candidate": name, "evidence_dir": str(stage.relative_to(out))})
        (evidence_root / "DELIVERY_CANDIDATES.json").write_text(
            json.dumps(evidence_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not rows:
        raise RuntimeError("No candidates were generated")
    if not any(row["selection_eligible"] for row in rows):
        for row in rows:
            row.pop("print_volume", None)
            for key in list(row):
                if key.startswith("_"):
                    row.pop(key)
        manifest = {
            "schema": "shadowforge_unified_v1",
            "status": "FAILED_ELIGIBILITY",
            "view_model": "fixed_two_wall",
            "n_views": 2,
            "branches": args.branch,
            "standards": {"min_iou": args.min_iou, "min_view_iou": args.min_view_iou,
                          "min_saving": args.min_saving, "max_saving": args.max_saving,
                          "min_erosion_survival": args.min_erosion_survival,
                          "selection_policy": args.selection_policy,
                          "max_unsupported_ratio": args.max_unsupported_ratio,
                          "max_support_islands": args.max_support_islands,
                          "min_bed_area_mm2": args.min_bed_area_mm2,
                          "screen": args.screen,
                          "structural_resolution": args.structural_resolution,
                          "sf_threshold": args.sf_threshold,
                          "pino_stop_margin": args.pino_stop_margin,
                          "pino_max_rollbacks": args.pino_max_rollbacks,
                          "allow_best_effort_fidelity_fallback":
                              args.allow_best_effort_fidelity_fallback},
            "inputs": [str(p.resolve()) for p in args.silhouettes],
            "soft": soft_info,
            "pino_structural_policy": ranked_fem_summary,
            "candidates": rows,
            "reason": "No candidate passed every active fidelity, fabrication, and structural gate; no STL was exported.",
        }
        (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), "status": "FAILED_ELIGIBILITY"}, ensure_ascii=False, indent=2))
        return
    chosen = select(rows, args)
    display = next(v for name, v in candidates if name == chosen["candidate"])
    display_stl = out / "selected_display_fixed.stl"
    print_stl = out / f"selected_print_{chosen['print_orientation']}.stl"
    mesh = export_stl_at_print_scale(display, display_stl, args.print_mm)
    print_mesh = export_stl_at_print_scale(chosen["print_volume"], print_stl, args.print_mm)
    np.save(out / "selected_display_voxels256.npy", display.astype(np.uint8))
    pred_l = tw.wall_axes_to_image(tw.tw.project_left_wall_x(display))
    pred_r = tw.wall_axes_to_image(tw.tw.project_right_wall_y(display))
    (out / "projections").mkdir(exist_ok=True)
    n2.save_mask(pred_l, out / "projections/left_wall_x_projection.png")
    n2.save_mask(pred_r, out / "projections/right_wall_y_projection.png")
    tw.tw.contact_sheet("selected", left_img, right_img, pred_l, pred_r, out / "input_vs_projection.png")
    display_delivery = None
    if args.display_fixture:
        from shadowforge_pipeline.display_delivery import create_display_delivery
        display_delivery = create_display_delivery(
            display_stl,
            out / "display_delivery",
            margin_mm=args.fixture_margin_mm,
            base_thickness_mm=args.fixture_base_thickness_mm,
            clearance_mm=args.fixture_clearance_mm,
            retainer_height_mm=args.fixture_retainer_height_mm,
        )
    for row in rows:
        row.pop("print_volume", None)
        for key in list(row):
            if key.startswith("_"):
                row.pop(key)
    chosen_out = {k: v for k, v in chosen.items() if k != "print_volume"}
    manifest = {
        "schema": "shadowforge_unified_v1",
        "status": (
            "SUCCESS_BEST_EFFORT_FIDELITY" if fallback_activated else "SUCCESS"
        ),
        "view_model": "fixed_two_wall",
        "n_views": 2,
        "branches": args.branch,
        "standards": {"min_iou": args.min_iou, "min_view_iou": args.min_view_iou, "min_saving": args.min_saving, "max_saving": args.max_saving, "min_erosion_survival": args.min_erosion_survival, "selection_policy": args.selection_policy, "max_unsupported_ratio": args.max_unsupported_ratio, "max_support_islands": args.max_support_islands, "min_bed_area_mm2": args.min_bed_area_mm2, "screen": args.screen, "structural_resolution": args.structural_resolution, "sf_threshold": args.sf_threshold, "pino_stop_margin": args.pino_stop_margin, "pino_max_rollbacks": args.pino_max_rollbacks, "allow_best_effort_fidelity_fallback": args.allow_best_effort_fidelity_fallback},
        "inputs": [str(p.resolve()) for p in args.silhouettes],
        "soft": soft_info,
        "pino_structural_policy": ranked_fem_summary,
        "selected": chosen_out,
        "display_stl": str(display_stl), "print_stl": str(print_stl),
        "mesh_watertight": bool(mesh.is_watertight),
        "display_mesh_max_dimension_mm": float(np.max(mesh.extents)),
        "print_mesh_max_dimension_mm": float(np.max(print_mesh.extents)),
        "display_delivery": display_delivery,
        "candidates": rows,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "selected": chosen_out, "display_stl": str(display_stl), "print_stl": str(print_stl)}, ensure_ascii=False, indent=2))


def _camera_matrix(azimuth_deg: float, elevation_deg: float, device: torch.device) -> torch.Tensor:
    """World-from-camera matrix matching ``renderer_voxel.render_orthographic``."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    cz, sz, cx, sx = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    return torch.tensor(
        [[cz, -sz, 0.0, 0.0], [cx * sz, cx * cz, -sx, 0.0], [sx * sz, sx * cz, cx, 0.0],
         [0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32
    )


def _generic_targets(train_dir: Path, resolution: int) -> list[np.ndarray]:
    paths = sorted(train_dir.glob("gt_view_*.png"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError("Canonical soft trainer did not export normalized target views")
    return [n2.load_mask(path, resolution).astype(bool) for path in paths]


def _canonical_xyz_from_trainer_zyx(volume_zyx: np.ndarray) -> np.ndarray:
    """Map the canonical trainer lattice [Z,Y,X] into fabrication [X,Y,Z]."""
    return np.flip(
        np.transpose(np.asarray(volume_zyx), (2, 1, 0)),
        axis=(0, 2),
    )


def _sadekar_tfs_view_spec(index: int) -> tuple[int, str]:
    """Return trainer-frame ray axis and image transform for one direct TFS view."""
    specs = (
        (0, "ud"),
        (2, "tudlr"),
        (1, "udlr"),
        (0, "udlr"),
        (2, "tlr"),
        (1, "ud"),
    )
    if index < 0 or index >= len(specs):
        raise ValueError(f"Direct Sadekar TFS supports at most six views, got index {index}")
    return specs[index]


def _transform_sadekar_tfs_target(target: np.ndarray, transform: str) -> np.ndarray:
    """Align a normalized target with the corresponding trainer-frame ray plane."""
    image = np.asarray(target, dtype=bool)
    if transform.startswith("t"):
        image = image.T
    if "ud" in transform:
        image = np.flipud(image)
    if "lr" in transform:
        image = np.fliplr(image)
    return image


def _generic_visual_hull(
    targets: list[np.ndarray],
    args: argparse.Namespace,
    resolution: int,
    soft_info: dict | None = None,
    preserve_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject every declared orthographic target and intersect the masks.

    This is a protocol envelope, never a candidate by itself.  Strict TFS
    delivery always uses the exact Sadekar camera specification recorded by
    the soft stage; generic orthographic delivery uses its declared cameras.
    """
    dev = torch.device(args.device)
    masks: list[np.ndarray] = []
    n_base = len(args.silhouettes)
    if args.tsf and _uses_sadekar_direct_tfs(soft_info):
        for index, target in enumerate(targets):
            axis, transform = _sadekar_tfs_view_spec(index)
            image = _transform_sadekar_tfs_target(target, transform)
            masks.append(np.repeat(np.expand_dims(image, axis=axis), resolution, axis=axis))
    elif args.tsf:
        raise RuntimeError(
            "Strict TFS delivery requires the exact Sadekar camera metadata "
            "emitted by the soft stage; fixed-axis fallback is disabled."
        )
    else:
        views = build_views(n_base, bool(args.mr), False, False)
        if len(views) != len(targets):
            raise RuntimeError(f"Protocol produced {len(views)} views for {len(targets)} targets")
        for image, (azimuth, elevation) in zip(targets, views):
            camera_volume = torch.from_numpy(
                np.repeat(image[:, :, None], resolution, axis=2).astype(np.float32)
            )[None, None].to(dev)
            inverse = torch.linalg.inv(_camera_matrix(azimuth, elevation, dev))[:3][None]
            grid = F.affine_grid(inverse, camera_volume.shape, align_corners=False)
            world = F.grid_sample(camera_volume, grid, mode="nearest", padding_mode="zeros", align_corners=False)
            masks.append(world[0, 0].detach().cpu().numpy() >= 0.5)
    hull_zyx = np.logical_and.reduce(masks)
    # Backprojection is assembled in renderer order [Z,Y,X], whereas every
    # fabrication operator in this module consumes canonical [X,Y,Z].
    hull = (
        _canonical_xyz_from_trainer_zyx(hull_zyx)
        if args.tsf and _uses_sadekar_direct_tfs(soft_info)
        else np.transpose(hull_zyx, (2, 1, 0))
    )
    if preserve_xyz is not None:
        preserve = np.asarray(preserve_xyz, dtype=bool)
        if preserve.shape != hull.shape:
            raise ValueError(
                f"Preserved soft proposal shape {preserve.shape} does not match hull {hull.shape}"
            )
        # The exact TFS renderer uses emission-absorption ray marching, so a
        # binary extrusion is only an approximate repair envelope. A proposal
        # already validated by that renderer must never be clipped by it.
        hull |= preserve
    if not hull.any():
        raise RuntimeError("Declared camera protocol produced an empty visual envelope")
    return hull


def _trainer_zyx_from_canonical_xyz(volume_xyz: np.ndarray) -> np.ndarray:
    """Invert the trainer-to-physical frame mapping used by ``canonical_voxel_soft``."""
    return np.transpose(np.flip(np.asarray(volume_xyz), axis=(0, 2)), (2, 1, 0))


def _uses_sadekar_direct_tfs(soft_info: dict | None) -> bool:
    metrics = (soft_info or {}).get("metrics", {})
    return bool(
        metrics.get("tsf_direct")
        and metrics.get("camera_protocol") == "sadekar_voxel_tfs_mirror_orthographic"
    )


def _generic_render(
    volume_xyz: np.ndarray,
    args: argparse.Namespace,
    soft_info: dict | None = None,
) -> list[np.ndarray]:
    """Binary renders under the exact declared TFS or generic orthographic protocol."""
    if _uses_sadekar_direct_tfs(soft_info):
        from sadekar_protocol import SadekarTFSProtocol

        metrics = soft_info["metrics"]
        camera = metrics.get("sadekar_camera", {})
        protocol = SadekarTFSProtocol(
            device=torch.device(args.device),
            image_size=int(volume_xyz.shape[0]),
            zdist=float(camera.get("zdist", 2.5)),
            volume_extent_world=float(camera.get("volume_extent_world", 1.7)),
            n_pts_per_ray=int(camera.get("n_pts_per_ray", 150)),
        )
        density_zyx = _trainer_zyx_from_canonical_xyz(volume_xyz)
        density = torch.from_numpy(
            np.ascontiguousarray(density_zyx, dtype=np.float32)
        )[None, None].to(args.device)
        # Match the canonical trainer's binary evaluation path. PyTorch3D's
        # emission-absorption marcher requires densities to remain below one.
        density = density * 0.999
        n_views = len(args.silhouettes) * (2 if args.mr else 1)
        with torch.no_grad():
            return [
                protocol.binary_mask(protocol.render(density, index))
                .detach().cpu().numpy().astype(bool)
                for index in range(n_views)
            ]

    density = torch.from_numpy(np.transpose(volume_xyz.astype(np.float32), (2, 1, 0)))[None, None]
    density = (density * 0.999).to(args.device)
    with torch.no_grad():
        if args.tsf:
            raise RuntimeError(
                "Strict TFS rendering requires exact Sadekar soft-stage metadata; "
                "the fixed-axis renderer fallback is disabled."
            )
        views = build_views(len(args.silhouettes), bool(args.mr), False, False)
        images = [render_orthographic(density, azimuth, elevation) for azimuth, elevation in views]
    return [(image >= 0.5).detach().cpu().numpy().astype(bool) for image in images]


def _restore_sadekar_direct_tfs_rays(
    volume_xyz: np.ndarray,
    hull_xyz: np.ndarray,
    guidance_xyz: np.ndarray,
    targets: list[np.ndarray],
) -> np.ndarray:
    """Restore missing direct-TFS rays in the trainer lattice used by rendering."""
    volume = _trainer_zyx_from_canonical_xyz(
        np.asarray(volume_xyz, dtype=bool)
    ).copy()
    hull = _trainer_zyx_from_canonical_xyz(np.asarray(hull_xyz, dtype=bool))
    guidance = _trainer_zyx_from_canonical_xyz(
        np.asarray(guidance_xyz, dtype=np.float32)
    )

    for index, declared_target in enumerate(targets):
        axis, transform = _sadekar_tfs_view_spec(index)
        target = _transform_sadekar_tfs_target(declared_target, transform)
        present = volume.any(axis=axis)
        admissible = hull.any(axis=axis)
        missing = target & ~present & admissible
        if not missing.any():
            continue
        scores = np.where(hull, guidance, -np.inf)
        winners = np.argmax(scores, axis=axis)
        selected = np.zeros_like(volume)
        np.put_along_axis(
            selected,
            np.expand_dims(winners, axis=axis),
            np.expand_dims(missing, axis=axis),
            axis=axis,
        )
        volume |= selected & hull

    return _canonical_xyz_from_trainer_zyx(volume)


def _restore_generic_camera_rays(
    volume_xyz: np.ndarray,
    hull_xyz: np.ndarray,
    guidance_xyz: np.ndarray,
    targets: list[np.ndarray],
    args: argparse.Namespace,
    rounds: int = 2,
) -> np.ndarray:
    """Restore generic orthographic rays through the exact renderer cameras."""
    device = torch.device(args.device)
    world = torch.from_numpy(
        np.transpose(np.asarray(volume_xyz, dtype=np.float32), (2, 1, 0))
    )[None, None].to(device)
    hull = torch.from_numpy(
        np.transpose(np.asarray(hull_xyz, dtype=np.float32), (2, 1, 0))
    )[None, None].to(device)
    guidance = torch.from_numpy(
        np.transpose(np.asarray(guidance_xyz, dtype=np.float32), (2, 1, 0))
    )[None, None].to(device)
    views = build_views(len(args.silhouettes), bool(args.mr), False, False)
    if len(views) != len(targets):
        raise RuntimeError(
            f"Protocol produced {len(views)} views for {len(targets)} targets"
        )

    with torch.no_grad():
        for _ in range(rounds):
            changed = False
            for target_np, (azimuth, elevation) in zip(targets, views):
                matrix = _camera_matrix(azimuth, elevation, device)[:3][None]
                grid = F.affine_grid(matrix, world.shape, align_corners=False)
                camera_occ = F.grid_sample(
                    world, grid, mode="nearest", padding_mode="zeros",
                    align_corners=False
                )
                camera_hull = F.grid_sample(
                    hull, grid, mode="nearest", padding_mode="zeros",
                    align_corners=False
                )
                camera_score = F.grid_sample(
                    guidance, grid, mode="bilinear", padding_mode="zeros",
                    align_corners=False
                )
                target = torch.from_numpy(
                    np.ascontiguousarray(target_np, dtype=np.bool_)
                ).to(device)
                present = camera_occ[0, 0].bool().any(dim=2)
                admissible = camera_hull[0, 0].bool().any(dim=2)
                missing = target & ~present & admissible
                if not bool(missing.any()):
                    continue

                scores = camera_score[0, 0].masked_fill(
                    ~camera_hull[0, 0].bool(), -torch.inf
                )
                winners = scores.argmax(dim=2, keepdim=True)
                selected_camera = torch.zeros_like(camera_occ)
                selected_camera[0, 0].scatter_(
                    2, winners, missing.unsqueeze(2).to(camera_occ.dtype)
                )
                inverse = torch.linalg.inv(
                    _camera_matrix(azimuth, elevation, device)
                )[:3][None]
                inverse_grid = F.affine_grid(
                    inverse, selected_camera.shape, align_corners=False
                )
                selected_world = F.grid_sample(
                    selected_camera, inverse_grid, mode="bilinear",
                    padding_mode="zeros", align_corners=False
                )
                # A one-voxel camera sample can otherwise disappear when the
                # rotated grid is resampled back into the world lattice.
                # Conservatively retain every touched world voxel; clipping to
                # the protocol hull prevents exterior silhouette leakage.
                selected_world = (selected_world > 1.0e-6).to(world.dtype)
                before = int(world.count_nonzero().item())
                world = torch.maximum(world, selected_world * hull)
                changed |= int(world.count_nonzero().item()) > before
            if not changed:
                break

    restored_zyx = world[0, 0].detach().cpu().numpy() >= 0.5
    return np.transpose(restored_zyx, (2, 1, 0))


def restore_declared_view_rays(
    volume_xyz: np.ndarray,
    hull_xyz: np.ndarray,
    guidance_xyz: np.ndarray,
    targets: list[np.ndarray],
    args: argparse.Namespace,
    soft_info: dict | None = None,
) -> np.ndarray:
    """Projection-aware repair shared by all declared camera protocols."""
    volume = np.asarray(volume_xyz, dtype=bool) & np.asarray(hull_xyz, dtype=bool)
    if args.tsf and _uses_sadekar_direct_tfs(soft_info):
        return _restore_sadekar_direct_tfs_rays(
            volume, hull_xyz, guidance_xyz, targets
        )
    if args.tsf:
        raise RuntimeError(
            "Strict TFS ray restoration requires exact Sadekar soft-stage metadata; "
            "the fixed-axis fallback is disabled."
        )
    return _restore_generic_camera_rays(
        volume, hull_xyz, guidance_xyz, targets, args
    )


def _generic_metrics(
    volume: np.ndarray,
    targets: list[np.ndarray],
    hull_voxels: int,
    args: argparse.Namespace,
    soft_info: dict | None = None,
) -> dict:
    renders = _generic_render(volume, args, soft_info)
    if len(renders) != len(targets):
        raise RuntimeError("Rendered view count does not match normalized targets")
    scores = []
    for pred, target in zip(renders, targets):
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        scores.append((inter / max(union, 1), 2.0 * inter / max(int(pred.sum() + target.sum()), 1)))
    components = int(label(np.asarray(volume, dtype=bool), structure=S6)[1])
    eroded = binary_erosion(np.asarray(volume, dtype=bool), structure=S6, iterations=1)
    erosion_survival = float(eroded.sum() / max(int(np.asarray(volume, dtype=bool).sum()), 1))
    watertight, mesh_parts, mesh_outer_parts = mesh_qa(volume)
    options = [(support_metrics(v, args.print_mm), name, v) for name, v in print_orientations(volume)]
    fdm, orientation, print_volume = min(options, key=lambda item: (item[0]["unsupported_ratio"], item[0]["significant_islands"], -item[0]["bed_area_mm2"]))
    return {
        "mean_iou": float(np.mean([item[0] for item in scores])),
        "min_iou": float(min(item[0] for item in scores)),
        "mean_dice": float(np.mean([item[1] for item in scores])),
        "view_ious": [float(item[0]) for item in scores],
        "view_dices": [float(item[1]) for item in scores],
        "saving": float(1.0 - np.asarray(volume, dtype=bool).sum() / max(hull_voxels, 1)),
        "components": components,
        "erosion_survival": erosion_survival,
        "watertight": watertight,
        "mesh_parts": mesh_parts,
        "mesh_outer_parts": mesh_outer_parts,
        "mesh_internal_parts": int(max(mesh_parts - mesh_outer_parts, 0)),
        "print_orientation": orientation,
        "fdm_proxy": fdm,
        "print_volume": print_volume,
        "renders": renders,
    }


def _generic_prescreen_metrics(
    volume: np.ndarray,
    targets: list[np.ndarray],
    hull_voxels: int,
    args: argparse.Namespace,
    soft_info: dict | None = None,
) -> dict:
    """Cheap C48 projection/topology metrics used before 256^3 delivery."""
    renders = _generic_render(volume, args, soft_info)
    if len(renders) != len(targets):
        raise RuntimeError("Prescreen render count does not match normalized targets")
    view_ious: list[float] = []
    view_dices: list[float] = []
    for pred, target in zip(renders, targets):
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        view_ious.append(float(inter / max(union, 1)))
        view_dices.append(
            float(2.0 * inter / max(int(pred.sum() + target.sum()), 1))
        )
    occupied = int(np.asarray(volume, dtype=bool).sum())
    return {
        "mean_iou": float(np.mean(view_ious)),
        "min_iou": float(min(view_ious)),
        "mean_dice": float(np.mean(view_dices)),
        "view_ious": view_ious,
        "view_dices": view_dices,
        "occupied_voxels": occupied,
        "saving": float(1.0 - occupied / max(hull_voxels, 1)),
        "components": int(label(np.asarray(volume, dtype=bool), structure=S6)[1]),
    }


def _shortlist_generic_delivery_candidates(
    candidates: list[tuple[str, np.ndarray]],
    targets: list[np.ndarray],
    hull_voxels: int,
    args: argparse.Namespace,
    soft_info: dict | None,
    out: Path,
) -> list[tuple[str, np.ndarray]]:
    """Choose C48 trajectory representatives for expensive 256^3 delivery.

    The shortlist is an execution optimization, not a new admission rule.
    Within each guarded-carving trajectory, it retains the lightest state in
    the configured coarse fidelity band. Representatives are then ordered by
    connectivity, mean/worst-view fidelity, and material use.
    """
    limit = int(args.delivery_shortlist_size)
    if limit <= 0 or len(candidates) <= limit:
        return candidates

    audit: list[dict] = []
    volumes: dict[str, np.ndarray] = {}
    for index, (name, core) in enumerate(candidates, start=1):
        metrics = _generic_prescreen_metrics(
            core, targets, hull_voxels, args, soft_info
        )
        trajectory_id, trajectory_step = trajectory_metadata(name)
        row = {
            "candidate": name,
            "trajectory_id": trajectory_id,
            "trajectory_step": int(trajectory_step),
            **metrics,
        }
        audit.append(row)
        volumes[name] = core
        print(json.dumps({
            "stage": "core_prescreen",
            "index": index,
            "total": len(candidates),
            "candidate": name,
            "mean_iou": row["mean_iou"],
            "min_iou": row["min_iou"],
            "saving": row["saving"],
            "components": row["components"],
        }), flush=True)

    representatives: list[dict] = []
    for trajectory_id in dict.fromkeys(row["trajectory_id"] for row in audit):
        trajectory = [
            row for row in audit if row["trajectory_id"] == trajectory_id
        ]
        connected = [row for row in trajectory if row["components"] == 1]
        pool = connected or trajectory
        local_best = max(row["mean_iou"] for row in pool)
        band = [
            row for row in pool
            if row["mean_iou"] >= local_best - args.delivery_prescreen_iou_tolerance
        ]
        representatives.append(max(
            band,
            key=lambda row: (
                row["saving"],
                row["min_iou"],
                row["mean_iou"],
            ),
        ))

    representatives.sort(key=lambda row: (
        row["components"] != 1,
        -row["mean_iou"],
        -row["min_iou"],
        -row["saving"],
    ))
    selected = representatives[:limit]

    # If fewer trajectories exist than requested, fill remaining slots with
    # globally strongest distinct states.
    selected_names = {row["candidate"] for row in selected}
    if len(selected) < limit:
        remaining = [
            row for row in audit if row["candidate"] not in selected_names
        ]
        remaining.sort(key=lambda row: (
            row["components"] != 1,
            -row["mean_iou"],
            -row["min_iou"],
            -row["saving"],
        ))
        selected.extend(remaining[:limit - len(selected)])
        selected_names = {row["candidate"] for row in selected}

    for row in audit:
        row["selected_for_delivery"] = row["candidate"] in selected_names
    (out / "core_delivery_prescreen.json").write_text(
        json.dumps({
            "schema": "pcraft_core_delivery_prescreen_v1",
            "candidate_count": len(candidates),
            "shortlist_size": len(selected),
            "iou_tolerance": float(args.delivery_prescreen_iou_tolerance),
            "selected": [row["candidate"] for row in selected],
            "rows": audit,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "core_prescreen_complete",
        "candidate_count": len(candidates),
        "shortlist_size": len(selected),
        "selected": [row["candidate"] for row in selected],
    }), flush=True)
    return [(row["candidate"], volumes[row["candidate"]]) for row in selected]


def _generic_seed_trajectory(
    guidance: np.ndarray,
    hull: np.ndarray,
    thresholds: tuple[float, ...],
    bridge_radius: int,
    bridge_passes: int,
    bridge: bool = True,
    targets: list[np.ndarray] | None = None,
    args: argparse.Namespace | None = None,
    soft_info: dict | None = None,
    restore_rays: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Protocol-independent soft-seed repair and 6-connected bridge trajectories."""
    from voxel_ops import connect6_min_bridges
    result: list[tuple[str, np.ndarray]] = []
    envelope = hull
    for threshold in thresholds:
        seed = (guidance >= threshold) & envelope
        if not seed.any():
            continue
        if restore_rays and targets is not None and args is not None:
            seed = restore_declared_view_rays(
                seed, envelope, guidance, targets, args, soft_info
            )
        if bridge:
            repaired, _ = connect6_min_bridges(
                seed, envelope, max_passes=bridge_passes, bridge_radius=bridge_radius
            )
            if restore_rays and targets is not None and args is not None:
                repaired = restore_declared_view_rays(
                    repaired, envelope, guidance, targets, args, soft_info
                )
            label_name = f"soft_seed_t{threshold:.2f}:bridge6r{bridge_radius}"
        else:
            repaired = seed
            label_name = f"soft_seed_t{threshold:.2f}:raw"
        result.append((label_name, repaired.astype(bool)))
    if not result:
        raise RuntimeError("No nonempty soft seed remained inside the protocol envelope")
    return result


def _generic_tfs_carve_trajectory(
    scaffold: np.ndarray,
    guidance: np.ndarray,
    hull: np.ndarray,
    label_name: str,
    num_axes: int,
    bridge_radius: int,
    bridge_passes: int,
    checkpoint_voxels: int = 0,
    max_checkpoints: int = 256,
) -> list[tuple[str, np.ndarray]]:
    """Build a projection-guarded material trajectory for strict TFS views."""
    from voxel_ops import connect6_min_bridges

    trajectory = [(f"{label_name}:repaired", np.asarray(scaffold, dtype=bool).copy())]
    current = np.asarray(scaffold, dtype=bool).copy()
    if checkpoint_voxels > 0:
        for checkpoint in range(1, max_checkpoints + 1):
            previous = current.copy()
            current, removed = axis_guarded_carve_step(
                current,
                priority=guidance,
                num_axes=num_axes,
                max_removals=checkpoint_voxels,
            )
            current, _ = connect6_min_bridges(
                current,
                hull,
                max_passes=bridge_passes,
                bridge_radius=bridge_radius,
            )
            changed = int(np.count_nonzero(previous ^ current))
            if removed == 0 or changed == 0:
                print(json.dumps({
                    "stage": "geometry_carve_stop",
                    "protocol": f"strict_tfs_{num_axes}axis",
                    "reason": "no_further_net_removal",
                    "checkpoint": checkpoint,
                    "attempted_removals": int(removed),
                }), flush=True)
                break
            name = (
                f"{label_name}:carve_phase_{checkpoint}"
                f"_removed_{removed}_net_{changed}"
            )
            print(json.dumps({
                "stage": "geometry_carve",
                "protocol": f"strict_tfs_{num_axes}axis",
                "checkpoint": checkpoint,
                "candidate": name,
                "removed_voxels": int(removed),
                "net_changed_voxels": changed,
                "occupied_voxels": int(current.sum()),
            }), flush=True)
            trajectory.append((name, current.copy()))
        else:
            print(json.dumps({
                "stage": "geometry_carve_stop",
                "protocol": f"strict_tfs_{num_axes}axis",
                "reason": "max_checkpoint_guard",
                "max_checkpoints": int(max_checkpoints),
            }), flush=True)
        return trajectory

    for phase, rounds in enumerate((1, 2, 4, 8, 16, 32), start=1):
        removed_total = 0
        for _ in range(rounds):
            current, removed = axis_guarded_carve_step(
                current, priority=guidance, num_axes=num_axes
            )
            removed_total += removed
            if not removed:
                break
        current, _ = connect6_min_bridges(
            current,
            hull,
            max_passes=bridge_passes,
            bridge_radius=bridge_radius,
        )
        name = f"{label_name}:carve_phase_{phase}_removed_{removed_total}"
        print(json.dumps({
            "stage": "geometry_carve",
            "protocol": f"strict_tfs_{num_axes}axis",
            "phase": phase,
            "candidate": name,
            "removed_voxels": int(removed_total),
            "occupied_voxels": int(current.sum()),
        }), flush=True)
        trajectory.append((name, current.copy()))
        if not removed_total:
            break
    return trajectory


def _generic_camera_carve_trajectory(
    scaffold: np.ndarray,
    guidance: np.ndarray,
    hull: np.ndarray,
    targets: list[np.ndarray],
    args: argparse.Namespace,
    label_name: str,
    soft_info: dict | None = None,
    checkpoint_voxels: int = 0,
    max_checkpoints: int = 256,
) -> list[tuple[str, np.ndarray]]:
    """Build a projection-restored trajectory for arbitrary cameras.

    Generic orthographic rays are not axis-aligned in the voxel lattice. Each
    checkpoint therefore removes low-guidance simple boundary voxels, restores
    missing declared-view rays with the exact cameras, and reconnects the
    result inside the common projection envelope.
    """
    from voxel_ops import connect6_min_bridges

    current = np.asarray(scaffold, dtype=bool).copy()
    trajectory = [(f"{label_name}:repaired", current.copy())]
    removal_budget = checkpoint_voxels if checkpoint_voxels > 0 else max(
        32, int(round(current.sum() * 0.01))
    )

    for checkpoint in range(1, max_checkpoints + 1):
        previous = current.copy()
        pad = np.pad(current, 1)
        interior = (
            pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1]
            & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
            & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
        )
        boundary = np.argwhere(current & ~interior)
        if not len(boundary):
            break
        scores = guidance[boundary[:, 0], boundary[:, 1], boundary[:, 2]]
        boundary = boundary[np.argsort(scores, kind="stable")]

        removed = 0
        for x, y, z in boundary:
            if not current[x, y, z]:
                continue
            if not local_simple_point(current, int(x), int(y), int(z)):
                continue
            current[x, y, z] = False
            removed += 1
            if removed >= removal_budget:
                break
        if removed == 0:
            break

        current = restore_declared_view_rays(
            current, hull, guidance, targets, args, soft_info
        )
        current, _ = connect6_min_bridges(
            current,
            hull,
            max_passes=args.bridge_passes,
            bridge_radius=args.bridge_radius,
        )
        current = restore_declared_view_rays(
            current, hull, guidance, targets, args, soft_info
        )
        net_removed = int(previous.sum()) - int(current.sum())
        if net_removed <= 0:
            current = previous
            print(json.dumps({
                "stage": "geometry_carve_stop",
                "protocol": "generic_orthographic",
                "reason": "no_further_net_removal",
                "checkpoint": checkpoint,
                "attempted_removals": int(removed),
            }), flush=True)
            break

        name = (
            f"{label_name}:carve_phase_{checkpoint}"
            f"_removed_{removed}_net_{net_removed}"
        )
        print(json.dumps({
            "stage": "geometry_carve",
            "protocol": "generic_orthographic",
            "checkpoint": checkpoint,
            "candidate": name,
            "removed_voxels": int(removed),
            "net_removed_voxels": int(net_removed),
            "occupied_voxels": int(current.sum()),
        }), flush=True)
        trajectory.append((name, current.copy()))

    return trajectory


def _generic_connected_delivery(
    core: np.ndarray,
    hull_256: np.ndarray,
    shell: np.ndarray,
    include_shell: bool,
    guidance: np.ndarray | None = None,
    targets: list[np.ndarray] | None = None,
    args: argparse.Namespace | None = None,
    soft_info: dict | None = None,
    restore_rays: bool = True,
) -> np.ndarray:
    """Build one connected high-resolution delivery from repaired geometry.

    A legacy C48 core is lifted once. Native V256 geometry is preserved at its
    own resolution. In both cases, attach only the local projection shell,
    restore declared rays once, and run a bounded delivery-resolution repair
    only when restoration has introduced multiple components.
    """
    from voxel_ops import connect6_min_bridges

    hull_256 = np.asarray(hull_256, dtype=bool)
    if len(set(hull_256.shape)) != 1:
        raise ValueError(f"Generic delivery hull must be cubic, got {hull_256.shape}")
    geometry = np.asarray(core, dtype=bool)
    if geometry.ndim != 3 or len(set(geometry.shape)) != 1:
        raise ValueError(
            f"Generic delivery geometry must be cubic, got {geometry.shape}"
        )
    if geometry.shape == hull_256.shape:
        lifted = geometry.copy()
    elif geometry.shape[0] < hull_256.shape[0]:
        lifted = upsample_nearest_to(geometry, hull_256.shape[0])
    else:
        raise ValueError(
            "Generic delivery geometry cannot exceed the delivery resolution: "
            f"{geometry.shape} versus {hull_256.shape}"
        )
    delivery = lifted & hull_256
    shell_voxels = 0
    if include_shell:
        shell = np.asarray(shell, dtype=bool)
        if bool(
            getattr(args, "full_envelope_delivery_shell", False)
            if args is not None else False
        ):
            # Compatibility replay for the verified July 22 fixed-three-view
            # route: retain the complete projection-envelope boundary.
            local_shell = shell
        else:
            shell_radius = max(
                1,
                int(getattr(args, "delivery_shell_voxels", 1)
                    if args is not None else 1),
            )
            local_shell = shell & binary_dilation(
                delivery,
                structure=S6,
                iterations=shell_radius,
            )
        shell_voxels = int(local_shell.sum())
        delivery |= local_shell
    if (
        restore_rays
        and guidance is not None
        and targets is not None
        and args is not None
    ):
        delivery = restore_declared_view_rays(
            delivery, hull_256, guidance, targets, args, soft_info
        )
    _, components_before = label(delivery, structure=S6)
    bridge_voxels = 0
    connected = delivery
    if components_before > 1:
        connected, bridge_voxels = connect6_min_bridges(
            delivery,
            hull_256,
            max_passes=int(
                getattr(args, "delivery_bridge_passes", 8)
                if args is not None else 8
            ),
            bridge_radius=1,
        )
    _, components_after = label(connected, structure=S6)
    print(json.dumps({
        "stage": "generic_delivery_connectivity",
        "components_before": int(components_before),
        "components_after": int(components_after),
        "local_shell_voxels": int(shell_voxels),
        "bridge_voxels": int(bridge_voxels),
        "max_bridge_passes": int(
            getattr(args, "delivery_bridge_passes", 8)
            if args is not None else 8
        ),
    }), flush=True)
    return connected.astype(bool)


def run_generic_delivery(args: argparse.Namespace) -> None:
    """Physical delivery adapter for every declared soft camera protocol."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    probability, proposal, soft_info = canonical_voxel_soft(args, out)
    if args.branch == "soft":
        (out / "MANIFEST.json").write_text(json.dumps({
            "schema": "shadowforge_unified_v2", "status": "SOFT_PROPOSAL_ONLY",
            "view_model": "strict_tfs" if args.tsf else "generic_orthographic",
            "n_input_views": len(args.silhouettes), "tsf": int(args.tsf), "mr": int(args.mr), "soft": soft_info,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), "status": "SOFT_PROPOSAL_ONLY"}, ensure_ascii=False, indent=2))
        return
    train_dir = Path(soft_info["artifact_dir"])
    targets_256 = _generic_targets(train_dir, 256)
    targets_core = _generic_targets(train_dir, args.core_resolution)
    hull_256 = _generic_visual_hull(
        targets_256,
        args,
        256,
        soft_info=soft_info,
        preserve_xyz=proposal,
    )
    probability_core = downsample_max_to(probability, args.core_resolution)
    proposal_core = downsample_max_to(
        proposal.astype(np.float32), args.core_resolution
    ) >= 0.5
    hull_core = _generic_visual_hull(
        targets_core,
        args,
        args.core_resolution,
        soft_info=soft_info,
        preserve_xyz=proposal_core,
    )
    # ``canonical_voxel_soft`` already returns physical [X,Y,Z]. Keep the soft
    # guidance in that frame so repair winners align with the delivery geometry.
    guidance_core = probability_core
    v256_geometry_owner = bool(getattr(args, "v256_geometry_owner", False))
    if v256_geometry_owner:
        geometry_resolution = 256
        guidance_geometry = probability
        hull_geometry = hull_256
        targets_geometry = targets_256
    else:
        geometry_resolution = int(args.core_resolution)
        guidance_geometry = guidance_core
        hull_geometry = hull_core
        targets_geometry = targets_core
    geometry_label = f"geometry{geometry_resolution}"
    targets_eval = targets_256
    candidates: list[tuple[str, np.ndarray]] = []
    if args.branch == "geometry":
        candidates.append(("geometry:protocol_hull", hull_geometry))
    else:
        repaired_seeds = _generic_seed_trajectory(
            guidance_geometry,
            hull_geometry,
            parse_soft_seed_thresholds(args.soft_seed_thresholds),
            args.bridge_radius,
            args.bridge_passes,
            bridge=not args.disable_bridge,
            targets=targets_geometry,
            args=args,
            soft_info=soft_info,
            restore_rays=not args.disable_ray_restoration,
        )
        if not args.disable_guarded_carve:
            for label_name, scaffold in repaired_seeds:
                if args.tsf:
                    candidates.extend(_generic_tfs_carve_trajectory(
                        scaffold,
                        guidance_geometry,
                        hull_geometry,
                        label_name,
                        len(args.silhouettes),
                        args.bridge_radius,
                        args.bridge_passes,
                        args.carve_checkpoint_voxels,
                        args.max_carve_checkpoints,
                    ))
                else:
                    candidates.extend(_generic_camera_carve_trajectory(
                        scaffold,
                        guidance_geometry,
                        hull_geometry,
                        targets_geometry,
                        args,
                        label_name,
                        soft_info,
                        args.carve_checkpoint_voxels,
                        args.max_carve_checkpoints,
                    ))
        else:
            candidates.extend(repaired_seeds)
    candidates = _shortlist_generic_delivery_candidates(
        candidates,
        targets_geometry,
        int(hull_geometry.sum()),
        args,
        soft_info,
        out,
    )
    rows: list[dict] = []
    structural_screen = StructuralScreen(args)
    full_envelope_shell = (
        hull_256
        & ~binary_erosion(
            hull_256,
            structure=S6,
            iterations=args.delivery_shell_voxels,
        )
    )
    delivered_candidates: dict[str, np.ndarray] = {}
    trajectory_gate_state: dict[str, dict[str, bool]] = {}
    for name, geometry in candidates:
        trajectory_id, trajectory_step = trajectory_metadata(name)
        gate_state = trajectory_gate_state.setdefault(
            trajectory_id, {"seen_pass": False, "stopped": False}
        )
        if gate_state["stopped"]:
            continue
        # Use the same full protocol-envelope shell as the verified two-wall
        # path.  A third TFS view adds one more Sadekar camera constraint; it
        # does not switch delivery into a separate fixed-axis shell policy.
        shell = full_envelope_shell
        delivered = _generic_connected_delivery(
            geometry,
            hull_256,
            shell,
            include_shell=not args.disable_delivery_shell,
            guidance=probability,
            targets=targets_256,
            args=args,
            soft_info=soft_info,
            restore_rays=not args.disable_ray_restoration,
        )
        delivered_candidates[name] = delivered
        data = _generic_metrics(
            delivered,
            targets_eval,
            int(hull_256.sum()),
            args,
            soft_info,
        )
        data.update({
            "candidate": f"{name}:{geometry_label}:delivery256",
            "geometry_owner": geometry_label,
            "eligible": False,
        })
        data["fabrication_pass"] = fabrication_eligible(data, args)
        data["fabrication_pass_without_fidelity"] = fabrication_eligible_without_fidelity(data, args)
        data["structural_screen_admissible"] = bool(
            data["fabrication_pass"]
            or (
                args.allow_best_effort_fidelity_fallback
                and data["fabrication_pass_without_fidelity"]
            )
        )
        if args.screen == "skip":
            safety = {"status": "NOT_REQUESTED", "safe": None, "decided_by": "none"}
        elif not data["structural_screen_admissible"]:
            safety = {"status": "NOT_EVALUATED", "safe": False, "decided_by": "fabrication_gate"}
        else:
            orientations = dict(print_orientations(hull_256))
            proxy, proxy_hull = structural_proxy(data["print_volume"], orientations[data["print_orientation"]], args.structural_resolution)
            data["_structural_volume"] = proxy
            data["_structural_hull"] = proxy_hull
            data["_structural_material"] = int(data["print_volume"].sum())
            data["_trajectory_id"] = trajectory_id
            data["_trajectory_step"] = trajectory_step
            safety = (
                {"status": "PENDING_PINO_POLICY", "safe": False,
                 "decided_by": "RAW_PINO_TRAJECTORY_POLICY"}
                if args.screen in {"pino_ranked_fem", "pino_rollback_fem"}
                else structural_screen.evaluate(proxy, proxy_hull)
            )
            safety["structural_resolution"] = int(args.structural_resolution)
            safety["structural_representation"] = "max-pooled delivered print orientation"
        data["structural_screen"] = safety
        data["structural_pass"] = bool(safety.get("safe", False))
        data["safety_score"] = float(safety.get("safety_score", safety.get("sf_fem", 0.0)) or 0.0)
        data["eligible"] = eligible(data, args)
        data["trajectory_step"] = int(trajectory_step)
        data["trajectory_gate_pass"] = bool(data["eligible"])
        if args.stop_at_first_gate_failure and args.screen not in {
            "pino_ranked_fem", "pino_rollback_fem"
        }:
            if data["eligible"]:
                gate_state["seen_pass"] = True
            elif gate_state["seen_pass"]:
                data["first_gate_failure"] = True
                gate_state["stopped"] = True
                print(json.dumps({
                    "stage": "trajectory_gate_stop",
                    "trajectory_id": trajectory_id,
                    "failed_step": int(trajectory_step),
                    "failed_candidate": data["candidate"],
                    "rollback_step": int(trajectory_step - 1),
                }), flush=True)
        rows.append(data)
        if args.screen not in {"pino_ranked_fem", "pino_rollback_fem"}:
            print(json.dumps({"stage": "protocol_delivery_qa", "candidate": data["candidate"], "mean_iou": data["mean_iou"],
                              "saving": data["saving"], "components": data["components"], "watertight": data["watertight"],
                              "eligible": data["eligible"]}), flush=True)
    ranked_fem_summary = None
    if args.screen == "pino_ranked_fem":
        ranked_fem_summary = apply_pino_ranked_fem(rows, structural_screen, args)
    elif args.screen == "pino_rollback_fem":
        ranked_fem_summary = apply_pino_rollback_fem(rows, structural_screen, args)
    if args.screen in {"pino_ranked_fem", "pino_rollback_fem"}:
        for data in rows:
            data["eligible"] = eligible(data, args)
            print(json.dumps({"stage": "protocol_delivery_qa", "candidate": data["candidate"],
                              "mean_iou": data["mean_iou"], "saving": data["saving"],
                              "pino_rank": data.get("pino_rank"),
                              "pino_trajectory_rank": data.get("pino_trajectory_rank"),
                              "pino_monotone_sf": data.get("pino_monotone_sf"),
                              "fem_checked": data.get("fem_checked", False),
                              "structural_status": data["structural_screen"]["status"],
                              "eligible": data["eligible"]}), flush=True)
    fallback_activated = mark_selection_eligibility(rows, args)
    if not any(row["selection_eligible"] for row in rows):
        diagnostic_export = None
        if args.export_best_rejected and rows:
            rejected = select(rows, args)
            rejected_volume = np.asarray(rejected["print_volume"], dtype=bool)
            rejected_dir = out / "best_rejected_candidate"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            volume_path = rejected_dir / "volume.npy"
            display_path = rejected_dir / "candidate_display_100mm.stl"
            np.save(volume_path, rejected_volume)
            mesh = export_stl_at_print_scale(
                rejected_volume, display_path, args.print_mm
            )
            projection_dir = rejected_dir / "projections"
            projection_dir.mkdir(exist_ok=True)
            for index, (target, pred) in enumerate(
                zip(targets_eval, rejected["renders"])
            ):
                n2.save_mask(
                    target, projection_dir / f"target_view_{index}.png"
                )
                n2.save_mask(
                    pred, projection_dir / f"projection_view_{index}.png"
                )
            diagnostic_export = {
                "status": "REJECTED_DIAGNOSTIC_ONLY",
                "candidate": rejected["candidate"],
                "reason": (
                    "This candidate did not pass every active delivery gate "
                    "and must not be reported as an eligible P-CRAFT result."
                ),
                "mean_iou": float(rejected["mean_iou"]),
                "min_iou": float(rejected["min_iou"]),
                "saving": float(rejected["saving"]),
                "voxel_components": int(rejected["components"]),
                "mesh_watertight": bool(mesh.is_watertight),
                "mesh_parts": int(len(mesh.split(only_watertight=False))),
                "volume_npy": str(volume_path),
                "display_stl": str(display_path),
                "projection_dir": str(projection_dir),
            }
            (rejected_dir / "QA.json").write_text(
                json.dumps(diagnostic_export, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        for row in rows:
            row.pop("print_volume", None)
            row.pop("renders", None)
            for key in list(row):
                if key.startswith("_"):
                    row.pop(key)
        manifest = {
            "schema": "shadowforge_unified_v2",
            "status": "FAILED_ELIGIBILITY",
            "view_model": "strict_tfs" if args.tsf else "generic_orthographic",
            "n_input_views": len(args.silhouettes),
            "n_declared_views": len(targets_eval),
            "tsf": int(args.tsf),
            "mr": int(args.mr),
            "delivery_protocol": "sadekar_direct_tfs" if args.tsf else "declared_camera_protocol",
            "geometry_owner": geometry_label,
            "delivery_shell_mode": f"{geometry_label}_attached_envelope_boundary",
            "branches": args.branch,
            "soft": soft_info,
            "pino_structural_policy": ranked_fem_summary,
            "candidates": rows,
            "diagnostic_export": diagnostic_export,
            "reason": (
                "No candidate passed all active non-fidelity, structural, and "
                "strict-or-enabled-best-effort selection gates."
            ),
        }
        (out / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(
            {"output": str(out), "status": "FAILED_ELIGIBILITY"},
            ensure_ascii=False,
            indent=2,
        ))
        return
    chosen = select(rows, args)
    selected_name = next(
        name for name, _ in candidates
        if f"{name}:{geometry_label}:delivery256" == chosen["candidate"]
    )
    selected_delivery = delivered_candidates[selected_name]
    display_stl = out / "selected_display_fixed.stl"
    print_stl = out / f"selected_print_{chosen['print_orientation']}.stl"
    mesh = export_stl_at_print_scale(selected_delivery, display_stl, args.print_mm)
    export_stl_at_print_scale(chosen["print_volume"], print_stl, args.print_mm)
    projection_dir = out / "projections"; projection_dir.mkdir(exist_ok=True)
    for index, (target, pred) in enumerate(zip(targets_eval, chosen.pop("renders"))):
        n2.save_mask(target, projection_dir / f"target_view_{index}.png")
        n2.save_mask(pred, projection_dir / f"projection_view_{index}.png")
    for row in rows:
        row.pop("print_volume", None); row.pop("renders", None)
        for key in list(row):
            if key.startswith("_"):
                row.pop(key)
    chosen_out = {k: v for k, v in chosen.items() if k not in {"print_volume", "renders"}}
    manifest = {
        "schema": "shadowforge_unified_v2",
        "status": (
            "SUCCESS_BEST_EFFORT_FIDELITY" if fallback_activated else "SUCCESS"
        ),
        "view_model": "strict_tfs" if args.tsf else "generic_orthographic", "n_input_views": len(args.silhouettes),
        "n_declared_views": len(targets_eval), "tsf": int(args.tsf), "mr": int(args.mr), "branches": args.branch,
        "delivery_protocol": "sadekar_direct_tfs" if args.tsf else "declared_camera_protocol",
        "geometry_owner": geometry_label,
        "delivery_shell_mode": f"{geometry_label}_attached_envelope_boundary",
        "soft": soft_info, "pino_structural_policy": ranked_fem_summary,
        "selected": chosen_out, "display_stl": str(display_stl), "print_stl": str(print_stl),
        "mesh_watertight": bool(mesh.is_watertight), "candidates": rows,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "status": manifest["status"], "selected": chosen_out,
                      "display_stl": str(display_stl), "print_stl": str(print_stl)}, ensure_ascii=False, indent=2))


def run_generic_soft_proposal(args: argparse.Namespace) -> None:
    """Compatibility wrapper for proposal-only invocations."""
    run_generic_delivery(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silhouettes", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--branch", choices=("geometry", "soft", "soft-repair", "soft-geometry"), required=True)
    ap.add_argument("--min-iou", "--min_iou", dest="min_iou", type=float, default=0.975)
    ap.add_argument("--min-view-iou", "--min_view_iou", dest="min_view_iou", type=float, default=0.95)
    ap.add_argument("--min-saving", "--min_saving", dest="min_saving", type=float, default=0.30)
    ap.add_argument("--max-saving", "--max_saving", dest="max_saving", type=float, default=0.85)
    ap.add_argument("--min-erosion-survival", "--min_erosion_survival", dest="min_erosion_survival", type=float, default=0.08)
    ap.add_argument("--max-unsupported-ratio", "--max_unsupported_ratio", dest="max_unsupported_ratio", type=float, default=1.0,
                    help="FDM preparation proxy gate; 1.0 disables this optional display-ranking gate.")
    ap.add_argument("--max-support-islands", "--max_support_islands", dest="max_support_islands", type=int, default=10**9,
                    help="Maximum significant unsupported islands in the selected print orientation.")
    ap.add_argument("--min-bed-area-mm2", "--min_bed_area_mm2", dest="min_bed_area_mm2", type=float, default=0.0,
                    help="Minimum model-to-bed contact area in mm^2 for the selected print orientation.")
    ap.add_argument("--selection-iou-tolerance", "--selection_iou_tolerance", dest="selection_iou_tolerance", type=float, default=0.01)
    ap.add_argument("--selection-policy", "--selection_policy", dest="selection_policy", choices=("material", "fdm"), default="material",
                    help="material preserves the paper selector; fdm ranks eligible physical-display candidates by support proxy.")
    ap.add_argument(
        "--allow-best-effort-fidelity-fallback",
        "--allow_best_effort_fidelity_fallback",
        dest="allow_best_effort_fidelity_fallback",
        action="store_true",
        help=(
            "If and only if no strict candidate exists, relax mean/worst-view "
            "IoU while retaining every material, topology, mesh, support, and "
            "structural gate. The manifest is marked SUCCESS_BEST_EFFORT_FIDELITY."
        ),
    )
    ap.add_argument("--print-mm", "--print_mm", dest="print_mm", type=float, default=100.0)
    ap.add_argument(
        "--screen",
        choices=("skip", "fem_only", "pino_ranked_fem", "pino_rollback_fem",
                 "pino_acs_fem", "v29_ensemble_acs_fem"),
        default="skip",
        help="Candidate structural gate shared with run.py. skip keeps the non-load-bearing protocol.",
    )
    ap.add_argument("--structural-resolution", "--structural_resolution", dest="structural_resolution", type=int, choices=(48,), default=48,
                    help="Resolution of the conservative delivered structural proxy.")
    ap.add_argument("--sf-threshold", "--sf_threshold", dest="sf_threshold", type=float, default=1.3)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Legacy ensemble significance level; frozen dual-side ACS reads its own safe/unsafe alphas.")
    ap.add_argument("--load-multiplier", "--load_multiplier", dest="load_multiplier", type=float, default=1.0)
    ap.add_argument("--pino-top-k", "--pino_top_k", dest="pino_top_k", type=int, default=3,
                    help="Initial Raw-PINO-ranked candidate prefix verified by voxel FEM.")
    ap.add_argument("--pino-top-k-fallback", "--pino_top_k_fallback", dest="pino_top_k_fallback",
                    choices=("none", "expand", "full"), default="expand",
                    help="If the initial prefix has no FEM-safe candidate, stop, expand by k, or verify all.")
    ap.add_argument("--pino-stop-margin", "--pino_stop_margin", dest="pino_stop_margin",
                    type=float, default=0.0,
                    help="Conservative PINO SF margin above --sf-threshold for trajectory stopping.")
    ap.add_argument("--pino-max-rollbacks", "--pino_max_rollbacks", dest="pino_max_rollbacks",
                    type=int, default=-1,
                    help="Maximum heavier-candidate FEM rollbacks; -1 checks until safe or exhausted.")
    ap.add_argument("--pino-ranked-policy", "--pino_ranked_policy", dest="pino_ranked_policy",
                    type=Path, default=None,
                    help="Calibration-frozen pino_ranked_fem_policy_v1; overrides --pino-top-k.")
    ap.add_argument("--pino-checkpoint", "--pino_checkpoint", dest="pino_checkpoint", type=Path,
                    default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model.pt")
    ap.add_argument(
        "--acs-policy", "--acs_policy", dest="acs_policy", type=Path, default=None,
        help="Frozen task-disjoint strict_pino_dualside_acs_v1 policy matching --pino-checkpoint.",
    )
    ap.add_argument("--display-fixture", "--display_fixture", dest="display_fixture", action="store_true",
                    help="Write a separate display plinth after selection. It is excluded from projection and paper metrics.")
    ap.add_argument("--fixture-margin-mm", "--fixture_margin_mm", dest="fixture_margin_mm", type=float, default=8.0)
    ap.add_argument("--fixture-base-thickness-mm", "--fixture_base_thickness_mm", dest="fixture_base_thickness_mm", type=float, default=4.0)
    ap.add_argument("--fixture-clearance-mm", "--fixture_clearance_mm", dest="fixture_clearance_mm", type=float, default=0.8)
    ap.add_argument("--fixture-retainer-height-mm", "--fixture_retainer_height_mm", dest="fixture_retainer_height_mm", type=float, default=1.5)
    ap.add_argument("--soft-resolution", "--soft_resolution", dest="soft_resolution", type=int, choices=(48, 256), default=48)
    ap.add_argument("--soft-control-res", "--soft_control_res", dest="soft_control_res", type=int, default=0,
                    help="Low-frequency control-grid resolution for the soft proposal; 0 uses independent logits.")
    ap.add_argument("--soft-proposal-mode", "--soft_proposal_mode", dest="soft_proposal_mode",
                    choices=("dense", "proposal", "regularized"), default="regularized",
                    help="dense is the Sadekar-compatible baseline; proposal is a projection-only control; "
                         "regularized is the ShadowForge soft occupancy stage.")
    ap.add_argument("--soft-disable-losses", "--soft_disable_losses", dest="soft_disable_losses", default="volume",
                    help="Comma-separated soft proposal losses disabled for a controlled ablation. "
                         "Material reduction is performed only by the downstream guarded carving trajectory.")
    ap.add_argument("--soft-connectivity-start-iou", "--soft_connectivity_start_iou",
                    dest="soft_connectivity_start_iou", type=float, default=0.60)
    ap.add_argument("--soft-fidelity-gate", "--soft_fidelity_gate",
                    dest="soft_fidelity_gate", type=float, default=0.85)
    ap.add_argument("--soft-structure-warmup-steps", "--soft_structure_warmup_steps",
                    dest="soft_structure_warmup_steps", type=int, default=300)
    ap.add_argument("--soft-structure-ramp-steps", "--soft_structure_ramp_steps",
                    dest="soft_structure_ramp_steps", type=int, default=300)
    ap.add_argument("--soft-guidance-threshold", "--soft_guidance_threshold", dest="soft_guidance_threshold", type=float, default=0.05,
                    help="Diagnostic threshold saved with the 48^3 guidance; it does not define the hybrid scaffold.")
    ap.add_argument("--soft-seed-thresholds", "--soft_seed_thresholds", dest="soft_seed_thresholds", default="0.15,0.25,0.35,0.45,0.55",
                    help="Fixed soft thresholds that seed repaired hard candidate trajectories.")
    ap.add_argument("--bridge-radius", "--bridge_radius", dest="bridge_radius", type=int, default=1,
                    help="Radius of envelope-contained 6-connected repair bridges.")
    ap.add_argument("--bridge-passes", "--bridge_passes", dest="bridge_passes", type=int, default=16,
                    help="Maximum bridge-repair passes for each soft-seeded candidate.")
    ap.add_argument("--soft-seed-only", "--soft_seed_only", dest="soft_seed_only", action="store_true",
                    help="Legacy compatibility flag. Soft-seeded candidate generation is now always enforced.")
    ap.add_argument("--save-candidate-evidence", "--save_candidate_evidence", dest="save_candidate_evidence", action="store_true",
                    help="Write real seed, repair, bridge, and delivery-candidate volumes plus two-view projections for figure construction.")
    ap.add_argument(
        "--export-best-rejected",
        "--export_best_rejected",
        dest="export_best_rejected",
        action="store_true",
        help=(
            "When strict eligibility fails, export the highest-ranked rejected "
            "delivery and its projections for diagnosis. The manifest keeps "
            "FAILED_ELIGIBILITY and marks the artifact as diagnostic only."
        ),
    )
    ap.add_argument("--core-resolution", "--core_resolution", dest="core_resolution", type=int, choices=(48,), default=48,
                    help="Structural geometry resolution after the completed soft proposal is transferred.")
    ap.add_argument(
        "--v256-geometry-owner",
        "--v256_geometry_owner",
        dest="v256_geometry_owner",
        action="store_true",
        help=(
            "Keep thresholding, ray restoration, bridge repair, and guarded "
            "carving at V256. C48 is then derived only as the PINO/FEM proxy "
            "and never expanded back to define the delivered surface."
        ),
    )
    ap.add_argument("--soft-iterations", "--soft_iterations", dest="soft_iterations", type=int, default=2000)
    ap.add_argument("--soft-lr", "--soft_lr", dest="soft_lr", type=float, default=0.007)
    ap.add_argument("--soft-save-every", "--soft_save_every", dest="soft_save_every", type=int, default=100)
    ap.add_argument("--soft-seed", "--soft_seed", dest="soft_seed", type=int, default=42,
                    help="Optimizer seed for repeatable soft-proposal studies.")
    ap.add_argument("--soft-export-final-step", "--soft_export_final_step", dest="soft_export_final_step", action="store_true",
                    help="For controlled soft-loss ablations, export the common final optimizer step instead of the best binary-projection checkpoint.")
    ap.add_argument("--soft-artifact-dir", "--soft_artifact_dir", dest="soft_artifact_dir", type=Path,
                    help="Reuse one frozen canonical soft artifact directory. Intended for downstream geometry ablations only.")
    ap.add_argument(
        "--soft-preserve-target-aspect",
        "--soft_preserve_target_aspect",
        dest="soft_preserve_target_aspect",
        action="store_true",
        help=(
            "Preserve each input silhouette's foreground width/height ratio "
            "during canonical soft preprocessing and audit the saved targets. "
            "Disabled by default for exact legacy replay."
        ),
    )
    ap.add_argument("--delivery-shell-voxels", "--delivery_shell_voxels", dest="delivery_shell_voxels", type=int, default=2,
                    help="Thickness of the 256^3 silhouette shell in soft-geometry delivery.")
    ap.add_argument(
        "--full-envelope-delivery-shell",
        "--full_envelope_delivery_shell",
        dest="full_envelope_delivery_shell",
        action="store_true",
        help=(
            "Compatibility replay for verified fixed-view runs that delivered "
            "the complete V256 projection-envelope boundary."
        ),
    )
    ap.add_argument("--delivery-bridge-passes", "--delivery_bridge_passes",
                    dest="delivery_bridge_passes", type=int, default=8,
                    help="Maximum number of 256^3 connectivity-repair passes after one delivery ray-restoration step.")
    ap.add_argument("--delivery-shortlist-size", "--delivery_shortlist_size",
                    dest="delivery_shortlist_size", type=int, default=5,
                    help="Maximum geometry-trajectory representatives promoted to delivery QA; "
                         "0 disables prescreen shortlisting.")
    ap.add_argument("--delivery-prescreen-iou-tolerance",
                    "--delivery_prescreen_iou_tolerance",
                    dest="delivery_prescreen_iou_tolerance", type=float, default=0.02,
                    help="Per-trajectory C48 fidelity band used only to choose the lightest delivery representative.")
    ap.add_argument("--disable-ray-restoration", "--disable_ray_restoration", dest="disable_ray_restoration", action="store_true",
                    help="Ablation only: do not restore missing target rays after thresholding or delivery.")
    ap.add_argument("--disable-bridge", "--disable_bridge", dest="disable_bridge", action="store_true",
                    help="Ablation only: do not add envelope-contained 6-connected bridges.")
    ap.add_argument("--disable-delivery-shell", "--disable_delivery_shell", dest="disable_delivery_shell", action="store_true",
                    help="Ablation only: deliver only the lifted 48^3 core without the 256^3 silhouette shell.")
    ap.add_argument("--disable-guarded-carve", "--disable_guarded_carve", dest="disable_guarded_carve", action="store_true",
                    help="Ablation only: stop after soft-seed repair and bridging; do not execute guarded material deletion.")
    ap.add_argument("--carve-checkpoint-voxels", "--carve_checkpoint_voxels",
                    dest="carve_checkpoint_voxels", type=int, default=0,
                    help="Maximum guarded deletions per saved trajectory checkpoint; "
                         "0 preserves the legacy exponentially spaced sweep schedule.")
    ap.add_argument("--max-carve-checkpoints", "--max_carve_checkpoints",
                    dest="max_carve_checkpoints", type=int, default=256,
                    help="Safety cap for fine-grained carving checkpoints; it is not "
                         "a material-selection stopping criterion.")
    ap.add_argument("--stop-at-first-gate-failure", "--stop_at_first_gate_failure",
                    dest="stop_at_first_gate_failure", action="store_true",
                    help="After a trajectory has passed all active gates, retain its "
                         "first failed boundary state and stop evaluating lighter states.")
    ap.add_argument("--tsf", type=int, choices=(0, 1), default=0,
                    help="Use mesh8's strict TFS camera preset (one to three orthogonal base views).")
    ap.add_argument("--mr", type=int, choices=(0, 1), default=0,
                    help="Add the source mesh pipeline's mirrored camera constraints.")
    ap.add_argument("--voxel-python", type=Path,
                    default=Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.bridge_radius < 1 or args.bridge_passes < 1:
        raise ValueError("--bridge-radius and --bridge-passes must both be positive")
    if args.carve_checkpoint_voxels < 0 or args.max_carve_checkpoints < 1:
        raise ValueError("--carve-checkpoint-voxels must be nonnegative and "
                         "--max-carve-checkpoints must be positive")
    if (args.delivery_shortlist_size < 0
            or args.delivery_bridge_passes < 1
            or args.delivery_prescreen_iou_tolerance < 0):
        raise ValueError("--delivery-shortlist-size and "
                         "--delivery-prescreen-iou-tolerance must be nonnegative, "
                         "and --delivery-bridge-passes must be positive")
    if args.stop_at_first_gate_failure and args.screen in {
        "pino_ranked_fem", "pino_rollback_fem"
    }:
        raise ValueError("--stop-at-first-gate-failure requires an immediate "
                         "screen such as fem_only; replay PINO policies from that "
                         "frozen gate-bounded trajectory.")
    if args.tsf == 1 and len(args.silhouettes) not in (1, 2, 3):
        raise ValueError("--tsf 1 supports one to three orthogonal base views; "
                         "use --tsf 0 for an arbitrary number of generic "
                         "orthographic views.")
    # Preserve the wall-aligned two-view adapter for the physical wall-corner
    # task. All other declared camera protocols use the generic backprojection
    # adapter, which applies the same seed, bridge, delivery, QA, and screen
    # contract without silently falling back to proposal-only output.
    if len(args.silhouettes) == 2 and args.tsf == 0 and args.mr == 0:
        run_two_wall(args)
    else:
        run_generic_delivery(args)


if __name__ == "__main__":
    main()
