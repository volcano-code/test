#!/usr/bin/env python3
"""Authoritative ShadowForge pipeline.

silhouettes -> free/fixed x geometry/soft candidates -> manufacturing gates ->
PINO+ACS with FEM fallback -> candidate selection -> 48-to-256 delivery -> STL QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "generation", ROOT / "experiments"):
    sys.path.insert(0, str(path))

from candidate_selection_batch import TAU, fem_sf, fit_acs, make_candidates, n6
from datasets import load_data_from_list
from run_all_baselines import load_pino
from structural.analysis.load_generator import LoadGenerator
from structural.training.pretrain_pino_v3 import FEMDatasetV3, compute_von_mises
from voxel_ops import (
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    evaluate_tfs_volume,
    resize_binary_volume,
    sdf_watertight_mesh,
)
from shadowforge_pipeline.v29_ensemble_screen import V29EnsembleScreen
from shadowforge_pipeline.pino_ranked_fem import verify_ranked_prefix, verify_trajectory_rollback

S6 = generate_binary_structure(3, 1)
SOFT_CANDIDATE_NAMES = {
    "learned_connected",
    "learned_printable",
    "learned_clip_clean",
}

PINO_YIELD_STRESS = 50.0e6


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def strict_pino_sf(
    model: torch.nn.Module,
    volume: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Run the exact six-channel preprocessing used by strict PINO training."""
    density_np = np.asarray(volume, dtype=np.float32)
    force_np = np.asarray(force, dtype=np.float32)
    fixed_np = np.asarray(fixed_mask, dtype=np.float32)
    if fixed_np.ndim == 3:
        fixed_np = fixed_np[None]
    if density_np.ndim != 3 or force_np.shape != (3, *density_np.shape):
        raise ValueError(
            f"Invalid strict PINO inputs: density={density_np.shape}, force={force_np.shape}"
        )
    if fixed_np.shape != (1, *density_np.shape):
        raise ValueError(
            f"Invalid strict PINO fixed mask: expected {(1, *density_np.shape)}, "
            f"got {fixed_np.shape}"
        )

    solid = density_np > FEMDatasetV3.SOLID_THRESHOLD
    if solid.any() and (~solid).any():
        sdf_np = (
            distance_transform_edt(solid) - distance_transform_edt(~solid)
        ).astype(np.float32) / (density_np.shape[0] / 2.0)
    else:
        sdf_np = np.zeros_like(density_np, dtype=np.float32)

    density = torch.from_numpy(density_np)[None, None].to(device)
    force_tensor = torch.from_numpy(force_np / FEMDatasetV3.FORCE_SCALE)[None].to(device)
    sdf = torch.from_numpy(sdf_np)[None, None].to(device)
    fixed = torch.from_numpy(fixed_np)[None].to(device)
    pino_input = torch.cat([density, force_tensor, sdf, fixed], dim=1)
    if pino_input.shape[1] != 6:
        raise RuntimeError(f"Strict PINO requires six physical channels, got {pino_input.shape[1]}")

    if not hasattr(model, "forward_with_scale"):
        raise TypeError("Strict PINO checkpoint has no independent scale branch.")
    fields, log_vm_max = model.forward_with_scale(pino_input)
    vm_normalized = compute_von_mises(fields[:, 3:].float())
    mask = density[0, 0] > FEMDatasetV3.SOLID_THRESHOLD
    vm_normalized_max = (
        float(vm_normalized[0, 0][mask].max().item()) if bool(mask.any()) else 1e-12
    )
    log_vm_max_value = float(log_vm_max[0, 0].item())
    vm_max = vm_normalized_max * (10.0 ** log_vm_max_value)
    sf = PINO_YIELD_STRESS / max(vm_max, 1e-12)
    return {
        "sf_pino": float(sf),
        "log10_sf_pino": float(math.log10(max(sf, 1e-12))),
        "vm_normalized_max": vm_normalized_max,
        "log10_vm_max_scale": log_vm_max_value,
        "vm_max_pino": float(vm_max),
        "yield_stress": PINO_YIELD_STRESS,
        "input_channels": int(pino_input.shape[1]),
    }


def erosion_survival(volume: np.ndarray) -> float:
    volume = np.asarray(volume, dtype=bool)
    return float(binary_erosion(volume, structure=S6).sum() / max(int(volume.sum()), 1))


def keep_largest_6(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=bool)
    labels, count = label(volume, structure=S6)
    if count <= 1:
        return volume
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def volume_hash(volume: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(volume, dtype=np.uint8).tobytes()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_masks(paths: list[Path], resolution: int) -> list[np.ndarray]:
    if len(paths) != 3:
        raise ValueError("ShadowForge currently requires exactly three silhouettes")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing silhouettes: {missing}")
    tensors, _ = load_data_from_list(
        [str(path) for path in paths],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    unique = [np.asarray(item).squeeze().astype(np.float32) for item in tensors]
    return unique + [np.fliplr(item) for item in unique]


def manufacturing_metrics(
    volume: np.ndarray,
    masks: list[np.ndarray],
    calibration: dict[str, Any],
    hull_voxels: int,
    image_size: int,
) -> dict[str, Any]:
    volume = keep_largest_6(volume)
    projection = evaluate_tfs_volume(volume, masks, calibration, image_size=image_size)
    try:
        mesh = sdf_watertight_mesh(volume, smooth_sigma=0.6)
        watertight = bool(mesh.is_watertight and mesh.is_winding_consistent)
        parts = mesh.split(only_watertight=False)
        mesh_parts = int(len(parts))
        if parts:
            global_min, global_max = mesh.bounds
            tol = 1e-3
            outer_parts = 0
            for part in parts:
                pmin, pmax = part.bounds
                touches_outer_bbox = bool(
                    np.any(np.isclose(pmin, global_min, atol=tol))
                    or np.any(np.isclose(pmax, global_max, atol=tol))
                )
                outer_parts += int(touches_outer_bbox)
            mesh_outer_parts = int(outer_parts)
            mesh_internal_parts = int(max(mesh_parts - mesh_outer_parts, 0))
        else:
            mesh_outer_parts = 0
            mesh_internal_parts = 0
    except Exception as exc:
        watertight = False
        mesh_parts = 0
        mesh_outer_parts = 0
        mesh_internal_parts = 0
        mesh_error = repr(exc)
    result = {
        "mean_iou": float(projection["mean_iou"]),
        "mean_dice": float(projection["mean_dice"]),
        "min_iou": float(projection["min_iou"]),
        "view_ious": [float(value) for value in projection["view_ious"]],
        "view_dices": [float(value) for value in projection["view_dices"]],
        "occupied_voxels": int(volume.sum()),
        "component_count_6": int(n6(volume)),
        "watertight": watertight,
        "mesh_parts": mesh_parts,
        "mesh_outer_parts": mesh_outer_parts,
        "mesh_internal_parts": mesh_internal_parts,
        "erosion_survival": erosion_survival(volume),
        "material_saved_vs_hull": float(1.0 - int(volume.sum()) / max(int(hull_voxels), 1)),
    }
    if not watertight and "mesh_error" in locals():
        result["mesh_error"] = mesh_error
    return result


def passes_manufacturing(row: dict[str, Any], args: argparse.Namespace) -> bool:
    mesh_ok = row["mesh_parts"] == 1
    if args.allow_internal_cavity_mesh_parts:
        mesh_ok = bool(row["mesh_outer_parts"] == 1 and row["component_count_6"] == 1)
    return bool(
        row["mean_iou"] >= args.min_iou
        and row["component_count_6"] == 1
        and row["watertight"]
        and mesh_ok
        and row["erosion_survival"] >= args.min_erosion_survival
        and args.min_saving <= row["material_saved_vs_hull"] <= args.max_saving
    )


def passes_core_candidate_gate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Gate low-resolution structural cores before 48-to-256 delivery.

    The decoupled pipeline intentionally does not require the 48^3 core to
    satisfy the final visual IoU.  Visual fidelity is checked after delivery,
    when the clipped upsampled core has been combined with the 256^3 shell.
    """
    return bool(
        row["component_count_6"] == 1
        and row["watertight"]
        and row["mesh_parts"] == 1
        and row["erosion_survival"] >= args.min_erosion_survival
        and args.min_saving <= row["material_saved_vs_hull"] <= args.max_saving
    )


def mark_selection(row: dict[str, Any], *, status: str, verified: bool, fallback: bool) -> dict[str, Any]:
    result = dict(row)
    result["_selection_status"] = status
    result["_selection_verified"] = bool(verified)
    result["_selection_fallback"] = bool(fallback)
    result["selection_status"] = status
    result["selection_verified"] = bool(verified)
    result["selection_fallback"] = bool(fallback)
    return result


def best_relative_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            bool(row.get("manufacturing_pass", False)),
            bool(row.get("core_candidate_pass", False)),
            float(row.get("mean_iou", 0.0)),
            int(row.get("component_count_6", 999999)) == 1,
            bool(row.get("watertight", False)),
            float(row.get("erosion_survival", 0.0)),
            float(row.get("material_saved_vs_hull", -1.0)),
            float(row.get("safety_score", 0.0)),
        ),
    )


def choose_candidate(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    eps_saving: float = 0.02,
    eps_iou: float = 0.02,
):
    feasible = [row for row in rows if row.get("manufacturing_pass") and row.get("safety_pass")]
    if not feasible:
        candidate = best_relative_candidate(rows)
        if candidate is None:
            return None
        return mark_selection(
            candidate,
            status="fallback_no_fully_verified_candidate",
            verified=False,
            fallback=True,
        )
    if args.selection_saving_target >= 0:
        target = float(args.selection_saving_target)
        best_dist = min(abs(row["material_saved_vs_hull"] - target) for row in feasible)
        near = [row for row in feasible if abs(row["material_saved_vs_hull"] - target) <= best_dist + eps_saving]
    else:
        best_saving = max(row["material_saved_vs_hull"] for row in feasible)
        near = [row for row in feasible if best_saving - row["material_saved_vs_hull"] < eps_saving]
    iou_cap = max(row["mean_iou"] for row in near)
    near = [row for row in near if row["mean_iou"] >= iou_cap - eps_iou]
    fixed = [row for row in near if row["branch"] == "fixed"]
    pool = fixed if fixed else near
    return mark_selection(
        max(pool, key=lambda row: (row.get("safety_score", 0.0), row["mean_iou"])),
        status="verified",
        verified=True,
        fallback=False,
    )


class StructuralScreen:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.load_generator = LoadGenerator()
        self.model = None
        self.iso = None
        self.q_safe = None
        self.q_unsafe = None
        self.acs_policy = None
        self.ranked_policy = None
        self.v29 = None
        if args.screen in {"pino_ranked_fem", "pino_rollback_fem"}:
            policy_path = (
                getattr(args, "pino_ranked_policy", None)
                if args.screen == "pino_ranked_fem" else None
            )
            if policy_path is not None:
                if not policy_path.is_file():
                    raise FileNotFoundError(f"Frozen PINO ranking policy not found: {policy_path}")
                policy = json.loads(policy_path.read_text())
                if policy.get("schema") != "pino_ranked_fem_policy_v1":
                    raise ValueError(f"Unsupported PINO ranking policy schema: {policy.get('schema')!r}")
                recorded_checkpoint = Path(policy["checkpoint"])
                if not recorded_checkpoint.is_absolute():
                    recorded_checkpoint = ROOT / recorded_checkpoint
                if recorded_checkpoint.resolve() != args.pino_checkpoint.resolve():
                    raise ValueError(
                        f"PINO ranking policy uses {recorded_checkpoint.resolve()}, "
                        f"but pipeline requested {args.pino_checkpoint.resolve()}"
                    )
                if not np.isclose(float(policy["threshold_sf"]), float(args.sf_threshold), atol=1e-9):
                    raise ValueError("PINO ranking policy and pipeline use different SF thresholds")
                args.pino_top_k = int(policy["top_k"])
                args.pino_top_k_fallback = str(policy["fallback"])
                self.ranked_policy = {**policy, "path": str(policy_path.resolve())}
            if not args.pino_checkpoint.is_file():
                raise FileNotFoundError(f"PINO checkpoint not found: {args.pino_checkpoint}")
            self.model = load_pino(str(args.pino_checkpoint), self.device)
        elif args.screen == "pino_acs_fem":
            if args.acs_policy is None or not args.acs_policy.is_file():
                raise FileNotFoundError(
                    "pino_acs_fem requires a frozen task-disjoint ACS policy: "
                    "pass --acs-policy <dualside_acs_results.json>"
                )
            policy_payload = json.loads(args.acs_policy.read_text())
            if policy_payload.get("schema") != "strict_pino_dualside_acs_v1":
                raise ValueError(
                    f"Unsupported ACS policy schema in {args.acs_policy}: "
                    f"{policy_payload.get('schema')!r}"
                )
            if not str(policy_payload.get("selection_protocol", "")).startswith("calibration only"):
                raise ValueError("ACS policy was not frozen using calibration-only selection.")
            recorded_checkpoint = Path(policy_payload["checkpoint"])
            if not recorded_checkpoint.is_absolute():
                recorded_checkpoint = ROOT / recorded_checkpoint
            if recorded_checkpoint.resolve() != args.pino_checkpoint.resolve():
                raise ValueError(
                    "PINO/ACS mismatch: policy checkpoint is "
                    f"{recorded_checkpoint.resolve()}, requested checkpoint is "
                    f"{args.pino_checkpoint.resolve()}"
                )
            policy_threshold = float(policy_payload["threshold_sf"])
            if not np.isclose(policy_threshold, float(args.sf_threshold), rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"PINO/ACS threshold mismatch: policy uses SF={policy_threshold}, "
                    f"pipeline requested SF={args.sf_threshold}"
                )
            selected = policy_payload["selected_policy"]
            self.q_safe = float(selected["q_safe_log_sf"])
            self.q_unsafe = float(selected["q_unsafe_log_sf"])
            self.acs_policy = {
                "path": str(args.acs_policy.resolve()),
                "schema": policy_payload["schema"],
                "checkpoint": str(recorded_checkpoint.resolve()),
                "threshold_sf": policy_threshold,
                "alpha_safe": float(selected["alpha_safe"]),
                "alpha_unsafe": float(selected["alpha_unsafe"]),
                "q_safe_log_sf": self.q_safe,
                "q_unsafe_log_sf": self.q_unsafe,
                "selection_protocol": policy_payload["selection_protocol"],
            }
            self.model = load_pino(str(args.pino_checkpoint), self.device)
        elif args.screen == "v29_ensemble_acs_fem":
            self.v29 = V29EnsembleScreen(
                ROOT,
                self.device,
                ROOT / "certification/v29_ensemble_candidate_frozen_20260703.json",
            )

    def evaluate(self, volume: np.ndarray, hull: np.ndarray) -> dict[str, Any]:
        volume = np.asarray(volume, dtype=bool)
        if self.args.screen == "skip":
            return {"status": "UNVERIFIED", "safe": False, "decided_by": "none"}
        if self.args.screen == "fem_only":
            result = fem_sf(volume, self.args.load_multiplier)
            sf = result["sf_fem"]
            return {
                **result,
                "status": "SAFE" if sf is not None and sf >= self.args.sf_threshold else "UNSAFE",
                "safe": bool(sf is not None and sf >= self.args.sf_threshold),
                "decided_by": "FEM",
            }

        load = self.load_generator.standing_load(volume.astype(np.float32))
        force = load["force"] * float(self.args.load_multiplier)
        fixed_mask = load["fixed_mask"]
        if self.args.screen in {"pino_ranked_fem", "pino_rollback_fem"}:
            _synchronize(self.device)
            started = time.perf_counter()
            pino_metrics = strict_pino_sf(
                self.model, volume.astype(np.float32), force, fixed_mask, self.device
            )
            _synchronize(self.device)
            elapsed_s = time.perf_counter() - started
            sf_pino = float(pino_metrics["sf_pino"])
            return {
                "status": "RANKED",
                "safe": None,
                "decided_by": "RAW_PINO_RANKING",
                "sf_pino": sf_pino,
                "safety_score": sf_pino,
                "elapsed_s": float(elapsed_s),
                "pino_diagnostics": pino_metrics,
            }
        if self.args.screen == "v29_ensemble_acs_fem":
            decision = self.v29.evaluate(volume.astype(np.float32), force, self.args.sf_threshold)
            if decision["needs_fem"]:
                fem = fem_sf(volume, self.args.load_multiplier)
                sf = fem["sf_fem"]
                return {
                    **decision,
                    **fem,
                    "status": "SAFE" if sf is not None and sf >= self.args.sf_threshold else "UNSAFE",
                    "safe": bool(sf is not None and sf >= self.args.sf_threshold),
                    "decided_by": "FEM_FALLBACK",
                    "safety_score": float(sf or 0.0),
                }
            return {
                **decision,
                "status": "SAFE" if decision["safe"] else "UNSAFE",
                "decided_by": "V29_ENSEMBLE_ACS",
                "safety_score": float(decision["acs_lower_sf"] if decision["safe"] else decision["acs_upper_sf"]),
            }

        pino_metrics = strict_pino_sf(
            self.model,
            volume.astype(np.float32),
            force,
            fixed_mask,
            self.device,
        )
        sf_pino = float(pino_metrics["sf_pino"])
        # The strict dual-side policy is calibrated directly on predicted
        # log10 safety factor.  No legacy isotonic map is mixed into deployment.
        pred = float(pino_metrics["log10_sf_pino"])
        lower = float(10 ** (pred - self.q_safe))
        upper = float(10 ** (pred + self.q_unsafe))
        if lower >= self.args.sf_threshold:
            return {
                "status": "SAFE",
                "safe": True,
                "decided_by": "ACS",
                "sf_pino": sf_pino,
                "acs_lower_sf": lower,
                "acs_upper_sf": upper,
                "safety_score": lower,
                "pino_diagnostics": pino_metrics,
                "acs_policy": self.acs_policy,
            }
        if upper < self.args.sf_threshold:
            return {
                "status": "UNSAFE",
                "safe": False,
                "decided_by": "ACS",
                "sf_pino": sf_pino,
                "acs_lower_sf": lower,
                "acs_upper_sf": upper,
                "safety_score": upper,
                "pino_diagnostics": pino_metrics,
                "acs_policy": self.acs_policy,
            }
        fem = fem_sf(volume, self.args.load_multiplier)
        sf = fem["sf_fem"]
        return {
            **fem,
            "status": "SAFE" if sf is not None and sf >= self.args.sf_threshold else "UNSAFE",
            "safe": bool(sf is not None and sf >= self.args.sf_threshold),
            "decided_by": "FEM_FALLBACK",
            "sf_pino": sf_pino,
            "acs_lower_sf": lower,
            "acs_upper_sf": upper,
            "safety_score": float(sf or 0.0),
            "pino_diagnostics": pino_metrics,
            "acs_policy": self.acs_policy,
        }

    def evaluate_fem(self, volume: np.ndarray) -> dict[str, Any]:
        result = fem_sf(np.asarray(volume, dtype=bool), self.args.load_multiplier)
        sf = result["sf_fem"]
        safe = bool(sf is not None and sf >= self.args.sf_threshold)
        return {
            **result,
            "status": "SAFE" if safe else "UNSAFE",
            "safe": safe,
            "decided_by": "FEM_SHORTLIST",
            "safety_score": float(sf or 0.0),
        }


def apply_pino_ranked_fem(
    rows: list[dict[str, Any]], screen: StructuralScreen, args: argparse.Namespace
) -> dict[str, Any]:
    """Rank valid candidates with Raw PINO and FEM-verify a short prefix."""
    rankable = []
    for row in rows:
        if not bool(row.get(
            "structural_screen_admissible",
            row.get("fabrication_pass", row.get("manufacturing_pass", False)),
        )):
            continue
        volume = row["_structural_volume"]
        hull = row["_structural_hull"]
        prediction = screen.evaluate(volume, hull)
        row["pino_prediction"] = prediction
        rankable.append({
            "candidate": row.get("candidate", row.get("candidate_id")),
            "sf_pino": float(prediction["sf_pino"]),
            "material": int(row.get("_structural_material", np.asarray(volume).sum())),
            "_row": row,
        })

    summary = verify_ranked_prefix(
        rankable,
        top_k=int(args.pino_top_k),
        threshold_sf=float(args.sf_threshold),
        fallback=args.pino_top_k_fallback,
        fem_evaluate=lambda item: screen.evaluate_fem(item["_row"]["_structural_volume"]),
    )
    summary["frozen_policy"] = screen.ranked_policy
    for item in rankable:
        row = item["_row"]
        row["pino_rank"] = item["pino_rank"]
        row["fem_checked"] = item["fem_checked"]
        if item["fem_checked"]:
            safety = {**item["fem_result"], "sf_pino": item["sf_pino"],
                      "pino_rank": item["pino_rank"]}
        else:
            safety = {
                "status": "NOT_FEM_CHECKED",
                "safe": False,
                "decided_by": "PINO_RANKED_OUTSIDE_SHORTLIST",
                "sf_pino": item["sf_pino"],
                "pino_rank": item["pino_rank"],
                "safety_score": item["sf_pino"],
            }
        row["structural_screen"] = safety
        row["structural_pass"] = bool(safety.get("safe", False))
        row["safety_pass"] = row["structural_pass"]
        row["safety_score"] = float(safety.get("safety_score", 0.0) or 0.0)
    return summary


def apply_pino_rollback_fem(
    rows: list[dict[str, Any]], screen: StructuralScreen, args: argparse.Namespace
) -> dict[str, Any]:
    """Let PINO locate each carving boundary and FEM verify with rollback."""
    rankable = []
    for row in rows:
        if not bool(row.get(
            "structural_screen_admissible",
            row.get("fabrication_pass", row.get("manufacturing_pass", False)),
        )):
            continue
        volume = row["_structural_volume"]
        hull = row["_structural_hull"]
        prediction = screen.evaluate(volume, hull)
        row["pino_prediction"] = prediction
        rankable.append({
            "candidate": row.get("candidate", row.get("candidate_id")),
            "sf_pino": float(prediction["sf_pino"]),
            "material": int(row.get("_structural_material", np.asarray(volume).sum())),
            "trajectory_id": row.get("_trajectory_id", "candidate_pool"),
            "trajectory_step": int(row.get("_trajectory_step", 0)),
            "_row": row,
        })

    summary = verify_trajectory_rollback(
        rankable,
        threshold_sf=float(args.sf_threshold),
        safety_margin=float(args.pino_stop_margin),
        max_rollbacks=int(args.pino_max_rollbacks),
        fem_evaluate=lambda item: screen.evaluate_fem(item["_row"]["_structural_volume"]),
    )
    for item in rankable:
        row = item["_row"]
        row["pino_trajectory_id"] = item["trajectory_id"]
        row["pino_trajectory_step"] = item["trajectory_step"]
        row["pino_trajectory_rank"] = item["trajectory_rank"]
        row["pino_monotone_sf"] = item["monotone_sf_pino"]
        row["pino_predicted_safe"] = item["pino_predicted_safe"]
        row["fem_checked"] = item["fem_checked"]
        if item["fem_checked"]:
            safety = {
                **item["fem_result"],
                "sf_pino": item["sf_pino"],
                "monotone_sf_pino": item["monotone_sf_pino"],
                "trajectory_rank": item["trajectory_rank"],
            }
        else:
            safety = {
                "status": "NOT_FEM_CHECKED",
                "safe": False,
                "decided_by": "PINO_TRAJECTORY_OUTSIDE_ROLLBACK_PATH",
                "sf_pino": item["sf_pino"],
                "monotone_sf_pino": item["monotone_sf_pino"],
                "trajectory_rank": item["trajectory_rank"],
                "safety_score": item["sf_pino"],
            }
        row["structural_screen"] = safety
        row["structural_pass"] = bool(safety.get("safe", False))
        row["safety_pass"] = row["structural_pass"]
        row["safety_score"] = float(safety.get("safety_score", 0.0) or 0.0)
    return summary


def run_soft_training(args: argparse.Namespace, branch: str, branch_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    train_dir = (branch_dir / "soft_training").resolve()
    candidate_dir = train_dir / "export_candidates"
    command = [
        str(args.python),
        "train.py",
        args.device,
        str(train_dir),
        str(args.soft_iters),
        str(args.soft_lr),
        "0",
        "-sdlist",
        *[str(path.resolve()) for path in args.silhouettes],
        "-mr", "1",
        "-tsf", "1",
        "-fixedcalib", "1" if branch == "fixed" else "0",
        "-res", str(args.soft_resolution),
        "-eres", str(args.cert_resolution),
        "-volmin", str(args.soft_vol_min),
        "--candidate_min_iou", str(args.min_iou),
        "--candidate_min_saving", str(args.min_saving),
        "--candidate_max_saving", str(args.max_saving),
        "--save_export_candidates", "1",
        "--renderer_phase4_iters", str(args.soft_renderer_phase4_iters),
        "--post_export_bundle", "0",
    ]
    signature = {
        "command": command,
        "input_sha256": {str(path.resolve()): file_hash(path) for path in args.silhouettes},
    }
    signature_path = train_dir / "pipeline_signature.json"
    cached = False
    if candidate_dir.is_dir() and signature_path.is_file():
        cached = json.loads(signature_path.read_text(encoding="utf-8")) == signature
    if train_dir.exists() and not cached:
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    meta = {**signature, "cached": cached}
    if not cached:
        with (train_dir / "pipeline_train.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT / "generation",
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": str(args.omp_threads)},
            )
        meta["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(f"soft training failed for {branch}; see {train_dir / 'pipeline_train.log'}")
    volumes = {}
    for path in sorted(candidate_dir.glob("*.npy")):
        if path.stem in SOFT_CANDIDATE_NAMES:
            volume = np.squeeze(np.load(path)) > 0.5
            volumes[path.stem] = resize_binary_volume(volume, args.cert_resolution)
    if not volumes:
        raise RuntimeError(f"no soft candidates found in {candidate_dir}")
    return volumes, meta


def decouple_volume(
    core: np.ndarray,
    hull_target: np.ndarray,
    target: int,
    shell_voxels: int,
    connect_iters: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    core_up = resize_binary_volume(np.asarray(core, dtype=bool), target)
    clipped = core_up & hull_target
    shell = hull_target & ~binary_erosion(hull_target, structure=S6, iterations=int(shell_voxels))
    volume = clipped | shell
    if connect_iters > 0:
        # The 256 shell may touch the upsampled core through very thin voxel
        # bridges.  Those bridges are 6-connected in the voxel grid but can be
        # split by SDF meshing.  Add material only inside the visual hull so the
        # prescribed silhouettes are preserved while shell islands are welded to
        # the structural core.
        support = binary_dilation(clipped, structure=S6, iterations=int(connect_iters)) & hull_target
        volume = volume | support
        volume = binary_dilation(volume, structure=S6, iterations=1) & hull_target
    if fill_holes:
        volume = binary_fill_holes(volume) & hull_target
    return keep_largest_6(volume)


def mesh_for_print(volume: np.ndarray, print_mm: float):
    mesh = sdf_watertight_mesh(np.asarray(volume, dtype=bool), smooth_sigma=0.6)
    scale = float(print_mm) / max(float(mesh.extents.max()), 1e-12)
    mesh.apply_scale(scale)
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])
    return mesh


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if not 0.0 <= args.min_saving <= args.max_saving < 1.0:
        raise ValueError("Require 0 <= min_saving <= max_saving < 1")
    if not 0.0 < args.min_iou <= 1.0:
        raise ValueError("min_iou must be in (0, 1]")
    if args.cert_resolution <= 0 or args.delivery_resolution < 0:
        raise ValueError("cert_resolution must be positive and delivery_resolution non-negative")
    if args.soft == "train" and not args.python.is_file():
        raise FileNotFoundError(f"Soft-training Python not found: {args.python}")
    if args.screen in {"pino_acs_fem", "pino_ranked_fem", "pino_rollback_fem"} and not args.pino_checkpoint.is_file():
        raise FileNotFoundError(f"PINO checkpoint not found: {args.pino_checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)
    masks = load_masks(args.silhouettes, args.cert_resolution)
    screen = StructuralScreen(args)
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    seen = set()

    for branch in args.branches:
        branch_dir = args.output / "branches" / branch
        candidate_dir = branch_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        geometry, aux = make_candidates(
            masks,
            args.cert_resolution,
            branch == "fixed",
            print_mm=args.print_mm,
            nozzle_mm=args.nozzle_mm,
            wall_safety=args.wall_safety,
        )
        calibration = aux["calibration"]
        hull = np.asarray(aux["hull"], dtype=bool)
        sources = {f"geometry:{name}": volume for name, volume in geometry.items()}
        soft_meta = None
        if args.soft == "train":
            soft, soft_meta = run_soft_training(args, branch, branch_dir)
            sources.update({f"soft:{name}": volume for name, volume in soft.items()})
        artifacts[branch] = {"soft": soft_meta, "n_raw_candidates": len(sources)}

        for source_name, raw_volume in sources.items():
            source, candidate_name = source_name.split(":", 1)
            volume = keep_largest_6(raw_volume)
            digest = (branch, volume_hash(volume))
            if digest in seen:
                continue
            seen.add(digest)
            path = candidate_dir / f"{source}_{candidate_name}.npy"
            np.save(path, volume.astype(np.uint8))
            row = {
                "candidate_id": f"{branch}:{source}:{candidate_name}",
                "branch": branch,
                "source": source,
                "candidate": candidate_name,
                "artifact": str(path),
                **manufacturing_metrics(volume, masks, calibration, int(hull.sum()), args.cert_resolution),
            }
            if args.delivery_resolution > 0:
                row["core_candidate_pass"] = passes_core_candidate_gate(row, args)
                row["core_iou_deferred_to_delivery"] = True
                row["manufacturing_pass"] = row["core_candidate_pass"]
            else:
                row["core_candidate_pass"] = passes_manufacturing(row, args)
                row["core_iou_deferred_to_delivery"] = False
                row["manufacturing_pass"] = row["core_candidate_pass"]
            row["_structural_volume"] = volume
            row["_structural_hull"] = hull
            row["_structural_material"] = int(volume.sum())
            row["_trajectory_id"] = f"{branch}:{source}"
            row["_trajectory_step"] = 0
            if row["manufacturing_pass"] and args.screen not in {"pino_ranked_fem", "pino_rollback_fem"}:
                try:
                    safety = screen.evaluate(volume, hull)
                except Exception as exc:
                    safety = {"status": "ERROR", "safe": False, "decided_by": "error", "error": repr(exc)}
            elif not row["manufacturing_pass"]:
                safety = {"status": "NOT_EVALUATED", "safe": False, "decided_by": "manufacturing_gate"}
            else:
                safety = {"status": "PENDING_PINO_POLICY", "safe": False,
                          "decided_by": "RAW_PINO_TRAJECTORY_POLICY"}
            row["structural_screen"] = safety
            row["safety_pass"] = bool(safety.get("safe", False))
            row["safety_score"] = float(safety.get("safety_score", safety.get("sf_fem", 0.0)) or 0.0)
            row["_volume"] = volume
            row["_calibration"] = calibration
            row["_hull"] = hull
            rows.append(row)

    ranked_fem_summary = None
    if args.screen == "pino_ranked_fem":
        ranked_fem_summary = apply_pino_ranked_fem(rows, screen, args)
    elif args.screen == "pino_rollback_fem":
        ranked_fem_summary = apply_pino_rollback_fem(rows, screen, args)

    selected = choose_candidate(rows, args)
    manifest: dict[str, Any] = {
        "schema_version": "shadowforge_authoritative_pipeline_v1",
        "status": "DEFER",
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_hash(path)}
            for path in args.silhouettes
        ],
        "config": {
            "branches": args.branches,
            "soft": args.soft,
            "soft_iters": args.soft_iters,
            "soft_lr": args.soft_lr,
            "soft_renderer_phase4_iters": args.soft_renderer_phase4_iters,
            "cert_resolution": args.cert_resolution,
            "delivery_resolution": args.delivery_resolution,
            "min_iou": args.min_iou,
            "saving_range": [args.min_saving, args.max_saving],
            "selection_saving_target": args.selection_saving_target,
            "delivery_connect_iters": args.delivery_connect_iters,
            "delivery_fill_holes": args.delivery_fill_holes,
            "allow_internal_cavity_mesh_parts": args.allow_internal_cavity_mesh_parts,
            "min_erosion_survival": args.min_erosion_survival,
            "core_gate": (
                "48^3 core gate excludes the final visual IoU when delivery_resolution>0; "
                "final visual IoU is checked on the delivered volume."
            ),
            "screen": args.screen,
            "sf_threshold": args.sf_threshold,
            "alpha": args.alpha,
            "pino_checkpoint": str(args.pino_checkpoint),
            "pino_top_k": args.pino_top_k,
            "pino_top_k_fallback": args.pino_top_k_fallback,
            "pino_stop_margin": args.pino_stop_margin,
            "pino_max_rollbacks": args.pino_max_rollbacks,
            "acs_calibration": str(
                ROOT / (
                    "certification/v29_ensemble_candidate_frozen_20260703.json"
                    if args.screen == "v29_ensemble_acs_fem"
                    else "certification/acs_candidate_recalibrated_20260703.json"
                )
            ),
        },
        "branch_artifacts": artifacts,
        "pino_structural_policy": ranked_fem_summary,
        "candidate_counts": {
            "total": len(rows),
            "manufacturing_pass": sum(bool(row["manufacturing_pass"]) for row in rows),
            "safety_pass": sum(bool(row["safety_pass"]) for row in rows),
        },
        "candidates": [serialize_row(row) for row in rows],
        "selected_core": serialize_row(selected) if selected else None,
    }
    if selected is None:
        manifest["defer_reason"] = "no candidate passed manufacturing and structural gates"
    else:
        core = selected["_volume"]
        calibration = selected["_calibration"]
        hull_low = selected["_hull"]
        if args.delivery_resolution > 0:
            target = int(args.delivery_resolution)
            masks_target = load_masks(args.silhouettes, target)
            hull_target, _ = build_tfs_visual_hull(
                masks_target,
                target,
                calibration=calibration,
            )
            shell_voxels = args.shell_voxels or int(math.ceil(
                args.wall_safety * args.nozzle_mm * target / args.print_mm
            ))
            delivery = decouple_volume(
                core,
                np.asarray(hull_target, dtype=bool),
                target,
                shell_voxels,
                args.delivery_connect_iters,
                args.delivery_fill_holes,
            )
            delivery_metrics = manufacturing_metrics(
                delivery,
                masks_target,
                calibration,
                int(np.asarray(hull_target).sum()),
                target,
            )
            delivery_metrics["shell_voxels"] = int(shell_voxels)
            delivery_metrics["manufacturing_pass"] = passes_manufacturing(delivery_metrics, args)
        else:
            delivery = core
            delivery_metrics = manufacturing_metrics(
                delivery,
                masks,
                calibration,
                int(hull_low.sum()),
                args.cert_resolution,
            )
            delivery_metrics["manufacturing_pass"] = passes_manufacturing(delivery_metrics, args)

        delivery_path = args.output / "selected_delivery.npy"
        np.save(delivery_path, delivery.astype(np.uint8))
        delivery_low = resize_binary_volume(delivery, args.cert_resolution)
        try:
            final_safety = (
                screen.evaluate_fem(delivery_low)
                if args.screen in {"pino_ranked_fem", "pino_rollback_fem"}
                else screen.evaluate(delivery_low, hull_low)
            )
        except Exception as exc:
            final_safety = {"status": "ERROR", "safe": False, "decided_by": "error", "error": repr(exc)}
        manifest["delivery"] = {
            "artifact": str(delivery_path),
            **delivery_metrics,
            "structural_recheck_at_resolution": args.cert_resolution,
            "structural_screen": final_safety,
        }
        if delivery_metrics["manufacturing_pass"] and final_safety.get("safe", False):
            mesh = mesh_for_print(delivery, args.print_mm)
            mesh_path = args.output / "shadowforge_selected.stl"
            mesh.export(mesh_path)
            manifest["delivery"]["mesh"] = {
                "artifact": str(mesh_path),
                "watertight": bool(mesh.is_watertight and mesh.is_winding_consistent),
                "parts": int(len(mesh.split(only_watertight=False))),
                "max_dimension_mm": float(mesh.extents.max()),
            }
            manifest["status"] = "PASS"
        else:
            manifest["defer_reason"] = "delivery failed manufacturing or structural recheck"

    manifest["runtime_seconds"] = float(time.time() - started)
    manifest_path = args.output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silhouettes", nargs=3, type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--branches", nargs="+", choices=["free", "fixed"], default=["free", "fixed"])
    ap.add_argument("--soft", choices=["train", "off"], default="train")
    ap.add_argument("--soft-iters", type=int, default=15000)
    ap.add_argument("--soft-lr", type=float, default=0.007)
    ap.add_argument("--soft-resolution", type=int, default=256)
    ap.add_argument("--soft-vol-min", type=float, default=0.08)
    ap.add_argument("--soft-renderer-phase4-iters", type=int, default=800)
    ap.add_argument("--cert-resolution", type=int, default=48)
    ap.add_argument("--delivery-resolution", type=int, default=256)
    ap.add_argument("--shell-voxels", type=int, default=0)
    ap.add_argument("--print-mm", type=float, default=100.0)
    ap.add_argument("--nozzle-mm", type=float, default=0.4)
    ap.add_argument("--wall-safety", type=float, default=2.0)
    ap.add_argument("--min-iou", type=float, default=0.95)
    ap.add_argument("--min-saving", type=float, default=0.40)
    ap.add_argument("--max-saving", type=float, default=0.85)
    ap.add_argument(
        "--selection-saving-target",
        type=float,
        default=-1.0,
        help=">=0 selects the verified candidate closest to this saving ratio instead of the lightest candidate.",
    )
    ap.add_argument(
        "--delivery-connect-iters",
        type=int,
        default=0,
        help="Add this many in-hull dilation iterations during 48-core/256-shell delivery to weld shell details to the core.",
    )
    ap.add_argument(
        "--delivery-fill-holes",
        action="store_true",
        help="Fill enclosed 3D voids after 48-core/256-shell delivery to remove internal mesh shells.",
    )
    ap.add_argument(
        "--allow-internal-cavity-mesh-parts",
        action="store_true",
        help="Allow watertight internal cavity shells when the voxel volume is one body and exactly one mesh component touches the outer boundary.",
    )
    ap.add_argument("--min-erosion-survival", type=float, default=0.15)
    ap.add_argument(
        "--screen",
        choices=["v29_ensemble_acs_fem", "pino_acs_fem", "pino_ranked_fem",
                 "pino_rollback_fem", "fem_only", "skip"],
        default="pino_acs_fem",
    )
    ap.add_argument("--sf-threshold", type=float, default=TAU)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--load-multiplier", type=float, default=1.0)
    ap.add_argument("--pino-top-k", type=int, default=3,
                    help="Initial Raw-PINO-ranked prefix verified by voxel FEM.")
    ap.add_argument("--pino-top-k-fallback", choices=("none", "expand", "full"), default="expand",
                    help="Policy when the initial FEM prefix contains no safe candidate.")
    ap.add_argument("--pino-stop-margin", type=float, default=0.0,
                    help="Conservative PINO SF margin above --sf-threshold for trajectory stopping.")
    ap.add_argument("--pino-max-rollbacks", type=int, default=-1,
                    help="Maximum heavier-candidate FEM rollbacks; -1 checks until safe or exhausted.")
    ap.add_argument("--pino-ranked-policy", type=Path, default=None,
                    help="Calibration-frozen pino_ranked_fem_policy_v1; overrides --pino-top-k.")
    ap.add_argument("--pino-checkpoint", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model.pt")
    ap.add_argument(
        "--acs-policy",
        type=Path,
        default=None,
        help="Frozen task-disjoint strict_pino_dualside_acs_v1 policy matching --pino-checkpoint.",
    )
    ap.add_argument("--python", type=Path, default=Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--omp-threads", type=int, default=8)
    return ap


def main() -> None:
    args = parser().parse_args()
    manifest = run(args)
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(args.output / "MANIFEST.json"),
        "selected_core": manifest.get("selected_core", {}).get("candidate_id") if manifest.get("selected_core") else None,
        "defer_reason": manifest.get("defer_reason"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
