#!/usr/bin/env python3
"""Project-local, fail-closed trainer for P-CRAFT soft-loss v2.

The executable optimizes only one continuous voxel-density field against the
fixed six-view silhouette protocol.  Its artifact allowlist intentionally
contains no geometry or mechanics formats.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_targets
from .losses import (
    LossProfile,
    SoftLossResult,
    aggregate_view_losses,
    boundary_weights,
    compact_soft_threshold,
    compute_soft_loss,
    registered_profiles,
    resolve_profile,
    robust_rho_schedule,
    threshold_margin_schedule,
)
from .metrics import evaluate_rendered_views
from .protocol import CAMERA_AZIMUTH_DEG, CAMERA_ELEVATION_DEG, SadekarTFSProtocol


DENSITY_THRESHOLDS = (0.04, 0.05, 0.06)
PRIMARY_DENSITY_THRESHOLD = 0.05
AUTHORITATIVE_SOURCE_ROOT = Path(__file__).resolve().parent
ALLOWED_ARTIFACT_PATTERNS = (
    "COMMAND.json",
    "STATUS.json",
    "run_manifest.json",
    "metrics.json",
    "history.jsonl",
    "checkpoint_latest.pt",
    "checkpoint_final.pt",
    "occupancy_probability.npy",
    "occupancy_binary_tau_*.npy",
    "gt_view_*.png",
    "continuous_view_*.png",
    "proxy_view_*.png",
    "binary_tau_*_view_*.png",
)


class ResumePreflightError(ValueError):
    """A resume request was rejected before any audit artifact was rewritten."""


@dataclass
class LegacyLossResult:
    """Exact decomposed legacy control loss, one value per registered view."""

    total: torch.Tensor
    per_view_total: torch.Tensor
    per_view_l2: torch.Tensor
    per_view_l1: torch.Tensor
    per_view_dice: torch.Tensor
    per_view_ssim: torch.Tensor
    per_view_false_positive: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "loss_l2": float(self.per_view_l2.detach().mean().cpu()),
            "loss_l1": float(self.per_view_l1.detach().mean().cpu()),
            "loss_dice": float(self.per_view_dice.detach().mean().cpu()),
            "loss_ssim": float(self.per_view_ssim.detach().mean().cpu()),
            "loss_false_positive": float(
                self.per_view_false_positive.detach().mean().cpu()
            ),
        }


def _fast_ssim_per_view(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("SSIM inputs must have matching [views,H,W] shapes")
    pred = prediction[:, None]
    truth = target.to(dtype=prediction.dtype)[:, None]
    mean_pred = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mean_truth = F.avg_pool2d(truth, window_size, 1, window_size // 2)
    variance_pred = (
        F.avg_pool2d(pred * pred, window_size, 1, window_size // 2)
        - mean_pred.square()
    )
    variance_truth = (
        F.avg_pool2d(truth * truth, window_size, 1, window_size // 2)
        - mean_truth.square()
    )
    covariance = (
        F.avg_pool2d(pred * truth, window_size, 1, window_size // 2)
        - mean_pred * mean_truth
    )
    ssim = (
        (2 * mean_pred * mean_truth + 0.01**2)
        * (2 * covariance + 0.03**2)
        / (
            (mean_pred.square() + mean_truth.square() + 0.01**2)
            * (variance_pred + variance_truth + 0.03**2)
            + 1e-8
        )
    )
    return 1.0 - ssim.mean(dim=(1, 2, 3))


def legacy_full_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    disable_l1: bool,
    disable_l2: bool,
    disable_ssim: bool,
) -> LegacyLossResult:
    """Frozen mean-view legacy control (never the historical dense 10x loss)."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("legacy loss inputs must have matching [views,H,W] shapes")
    truth = target.to(dtype=prediction.dtype)
    dims = (1, 2)
    per_l2 = (prediction - truth).square().mean(dim=dims)
    per_l1 = (prediction - truth).abs().mean(dim=dims)
    per_dice = 1.0 - (2.0 * (prediction * truth).sum(dim=dims) + 1.0) / (
        prediction.sum(dim=dims) + truth.sum(dim=dims) + 1.0
    )
    per_ssim = _fast_ssim_per_view(prediction, truth)
    per_false_positive = (prediction * (1.0 - truth)).mean(dim=dims)
    per_total = (
        (0.0 if disable_l2 else 1.0) * per_l2
        + (0.0 if disable_l1 else 0.35) * per_l1
        + 0.80 * per_dice
        + (0.0 if disable_ssim else 0.18) * per_ssim
        + 0.35 * per_false_positive
    )
    return LegacyLossResult(
        total=per_total.mean(),
        per_view_total=per_total,
        per_view_l2=per_l2,
        per_view_l1=per_l1,
        per_view_dice=per_dice,
        per_view_ssim=per_ssim,
        per_view_false_positive=per_false_positive,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P-CRAFT soft-stage loss-v2 trainer (soft-only, final-step export)"
    )
    parser.add_argument("device")
    parser.add_argument("output_dir")
    parser.add_argument("steps", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("exp_id", type=int)
    parser.add_argument("-sdlist", "--shadow-files", nargs=3, required=True)
    parser.add_argument("-mr", "--mirror-mode", type=int, default=1)
    parser.add_argument("-tsf", "--tsf-mode", type=int, default=1)
    parser.add_argument("-tsf4", "--tsf4-mode", type=int, default=0)
    parser.add_argument("-cam", "--camera-mode", default="ortho")
    parser.add_argument("-imsz", "--image-size", type=int, default=256)
    parser.add_argument("--res", type=int, choices=(128, 256), required=True)
    parser.add_argument(
        "-eres", "--export-res", dest="export_res", type=int, required=True
    )
    parser.add_argument("--control-res", type=int, default=0)
    parser.add_argument("--sadekar-zdist", type=float, default=1.7)
    parser.add_argument("--sadekar-volume-extent", type=float, default=1.7)
    parser.add_argument("--sadekar-npts-per-ray", type=int, default=150)
    parser.add_argument("--sadekar-density-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile", choices=registered_profiles(), required=True)
    parser.add_argument(
        "--view-aggregation", choices=("mean", "grouped_robust"), default=None
    )
    parser.add_argument("--disable-proxy", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--disable-boundary", action="store_true")
    parser.add_argument("--disable-l1", action="store_true")
    parser.add_argument("--disable-l2", action="store_true")
    parser.add_argument("--disable-ssim", action="store_true")
    parser.add_argument("--lambda-g", type=float, default=0.25)
    parser.add_argument("--lambda-b", type=float, default=0.10)
    parser.add_argument("--late-ssim-weight", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__OUTPUT_LATEST__",
        default=None,
        help="Resume the same resolved configuration; optional path defaults to output checkpoint_latest.pt.",
    )
    parser.add_argument(
        "--export-final-step",
        action="store_true",
        help="Accepted for compatibility; this trainer always exports the final step.",
    )
    parser.add_argument(
        "--soft-only-eval",
        action="store_true",
        required=True,
        help="Required fail-closed acknowledgement of the isolated experiment scope.",
    )
    return parser


def _equal_float(value: float, expected: float) -> bool:
    return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-9)


def validate_args(args: argparse.Namespace) -> None:
    """Reject every protocol drift from the pre-registered experiment."""

    if not bool(args.soft_only_eval):
        raise ValueError("--soft-only-eval is mandatory")
    if len(args.shadow_files) != 3:
        raise ValueError("exactly three base silhouettes are required")
    if args.mirror_mode != 1 or args.tsf_mode != 1 or args.tsf4_mode != 0:
        raise ValueError("fixed protocol requires -mr 1 -tsf 1 -tsf4 0")
    if args.camera_mode != "ortho" or args.image_size != 256:
        raise ValueError("fixed protocol requires -cam ortho -imsz 256")
    if args.res not in (128, 256):
        raise ValueError("--res must be 128 or 256")
    if args.export_res != args.res or args.control_res != 0:
        raise ValueError("-eres must equal --res and --control-res must be 0")
    if not _equal_float(args.sadekar_zdist, 1.7):
        raise ValueError("--sadekar-zdist must be 1.7")
    if not _equal_float(args.sadekar_volume_extent, 1.7):
        raise ValueError("--sadekar-volume-extent must be 1.7")
    if args.sadekar_npts_per_ray != 150:
        raise ValueError("--sadekar-npts-per-ray must be 150")
    if not _equal_float(args.sadekar_density_threshold, 0.05):
        raise ValueError("--sadekar-density-threshold must be 0.05")
    if args.seed != 17:
        raise ValueError("--seed must be 17")
    if args.steps <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.lambda_g < 0 or args.lambda_b < 0 or args.late_ssim_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if args.profile == "legacy_full" and args.view_aggregation not in (None, "mean"):
        raise ValueError("legacy_full is frozen to exact mean-view aggregation")
    if args.profile != "new_full" and args.late_ssim_weight != 0:
        raise ValueError("late SSIM is defined only for new_full")


def proxy_execution_semantics(args: argparse.Namespace) -> dict[str, Any]:
    """Describe whether the compact density/projection proxy is actually run."""

    is_new = args.profile == "new_full"
    threshold_enabled = is_new and not bool(args.disable_proxy)
    gate_or_boundary = is_new and (
        not bool(args.disable_gate) or not bool(args.disable_boundary)
    )
    return {
        "threshold_proxy_enabled": threshold_enabled,
        "proxy_blend_enabled": is_new and threshold_enabled,
        "auxiliary_proxy_required": gate_or_boundary and threshold_enabled,
        "auxiliary_domain": (
            "threshold_proxy"
            if gate_or_boundary and threshold_enabled
            else "raw_projection"
            if gate_or_boundary
            else "none"
        ),
    }


def training_auxiliary_projection(
    probability: torch.Tensor,
    raw_projection: torch.Tensor,
    *,
    args: argparse.Namespace,
    protocol: Any,
    progress: float,
) -> torch.Tensor | None:
    """Build Q only when enabled, otherwise route auxiliary terms to raw P."""

    if args.profile != "new_full":
        return None
    semantics = proxy_execution_semantics(args)
    if semantics["threshold_proxy_enabled"]:
        proxy_density = compact_soft_threshold(
            probability,
            tau=args.sadekar_density_threshold,
            margin=threshold_margin_schedule(progress),
        )
        return protocol.render_all(proxy_density * 0.999)
    if semantics["auxiliary_domain"] == "raw_projection":
        return raw_projection
    return None


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON, stable under mapping insertion order."""

    canonical = json.dumps(
        _json_normalize(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def implementation_source_manifest() -> dict[str, Any]:
    """Hash every project-local Python source that defines the soft trainer."""

    source_root = Path(AUTHORITATIVE_SOURCE_ROOT)
    sources = sorted(path for path in source_root.glob("*.py") if path.is_file())
    if not sources:
        raise RuntimeError(f"no authoritative soft trainer sources found in {source_root}")
    files_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    canonical = json.dumps(
        files_sha256, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        "algorithm": "sha256",
        "files_sha256": files_sha256,
        "bundle_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing shadow input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"output_dir", "resume", "export_final_step"}
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    shadow_files = [Path(path).resolve() for path in args.shadow_files]
    config["shadow_files"] = [str(path) for path in shadow_files]
    config["shadow_files_sha256"] = [_sha256_file(path) for path in shadow_files]
    config["density_thresholds"] = list(DENSITY_THRESHOLDS)
    config["primary_density_threshold"] = PRIMARY_DENSITY_THRESHOLD
    config["final_step_export_only"] = True
    config["implementation"] = implementation_source_manifest()
    return _json_normalize(config)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint_payload(
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    step: int,
    resolved_config: Mapping[str, Any],
    config_digest: str,
    last_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "pcraft_soft_loss_v2_checkpoint_v1",
        "logits": logits.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": _json_normalize(resolved_config),
        "config_hash": str(config_digest),
        "last_record": (
            None if last_record is None else _json_normalize(dict(last_record))
        ),
        "rng_state": _capture_rng_state(),
    }


def restore_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    expected_config: Mapping[str, Any],
    expected_digest: str,
) -> tuple[int, dict[str, Any] | None]:
    """Restore an exact training state and its last auditable step record."""

    _validate_checkpoint_metadata(payload, expected_config, expected_digest)
    with torch.no_grad():
        logits.copy_(payload["logits"].to(logits.device))
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng_state(payload["rng_state"])
    record = payload["last_record"]
    restored_record = None if record is None else dict(record)
    return int(payload["step"]) + 1, restored_record


def _validate_checkpoint_metadata(
    payload: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    expected_digest: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ResumePreflightError("resume checkpoint payload must be a mapping")
    required = {
        "logits",
        "optimizer",
        "step",
        "config",
        "config_hash",
        "last_record",
        "rng_state",
    }
    missing = required.difference(payload)
    if missing:
        raise ResumePreflightError(
            f"resume checkpoint is missing fields: {sorted(missing)}"
        )
    if payload["config_hash"] != expected_digest or payload["config"] != expected_config:
        raise ResumePreflightError(
            "resume checkpoint configuration does not match this run"
        )
    record = payload["last_record"]
    if record is not None and not isinstance(record, Mapping):
        raise ResumePreflightError(
            "resume checkpoint last_record must be a mapping or null"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_normalize(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _link_final_checkpoint(latest: Path, final: Path) -> None:
    temporary = final.with_name(f".{final.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(latest, temporary)
        os.replace(temporary, final)
    except OSError:
        if temporary.exists():
            temporary.unlink()
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        _atomic_checkpoint(final, payload)


def _append_history(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_normalize(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _profile_for_args(args: argparse.Namespace) -> LossProfile:
    profile = resolve_profile(
        args.profile,
        disable_proxy=args.disable_proxy,
        disable_gate=args.disable_gate,
        disable_boundary=args.disable_boundary,
        view_aggregation=args.view_aggregation,
    )
    return replace(
        profile,
        l1_weight=0.0 if args.disable_l1 else profile.l1_weight,
        l2_weight=0.0 if args.disable_l2 else profile.l2_weight,
    )


def _late_ssim_scale(progress: float, configured_weight: float) -> float:
    return float(configured_weight) * min(max((float(progress) - 0.75) / 0.25, 0.0), 1.0)


def _compute_step_loss(
    raw_projection: torch.Tensor,
    auxiliary_projection: torch.Tensor | None,
    targets: torch.Tensor,
    *,
    profile: LossProfile,
    args: argparse.Namespace,
    progress: float,
    cached_boundary_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if profile.legacy_terms:
        legacy = legacy_full_loss(
            raw_projection,
            targets,
            disable_l1=args.disable_l1,
            disable_l2=args.disable_l2,
            disable_ssim=args.disable_ssim,
        )
        return legacy.total, legacy.per_view_total, legacy.detached_scalars()

    result: SoftLossResult = compute_soft_loss(
        raw_projection,
        auxiliary_projection,
        targets,
        profile=profile,
        progress=progress,
        lambda_gate=args.lambda_g,
        lambda_boundary=args.lambda_b,
        cached_boundary_weights=cached_boundary_weights,
    )
    per_view_total = result.per_view_total
    ssim_scale = (
        0.0
        if args.disable_ssim
        else _late_ssim_scale(progress, args.late_ssim_weight)
    )
    per_view_ssim = _fast_ssim_per_view(raw_projection, targets)
    if ssim_scale:
        per_view_total = per_view_total + ssim_scale * per_view_ssim
        rho = (
            robust_rho_schedule(progress)
            if profile.view_aggregation == "grouped_robust"
            else 0.0
        )
        total = aggregate_view_losses(
            per_view_total,
            mode=profile.view_aggregation,
            beta=4.0,
            rho=rho,
        )
    else:
        total = result.total
    scalars = result.detached_scalars()
    scalars["loss"] = float(total.detach().cpu())
    scalars["loss_ssim"] = float(per_view_ssim.detach().mean().cpu())
    scalars["late_ssim_scale"] = float(ssim_scale)
    return total, per_view_total, scalars


def _format_threshold(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _write_image(path: Path, image: torch.Tensor, *, binary: bool = False) -> None:
    array = image.detach().float().cpu().numpy()
    if binary:
        output = array.astype(bool).astype(np.uint8) * 255
    else:
        output = np.clip(array * 255.0, 0.0, 255.0).round().astype(np.uint8)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"failed to write image {path}")


def _assert_artifact_allowlist(output: Path) -> None:
    unexpected = [
        path.name
        for path in output.iterdir()
        if path.is_file()
        and not any(fnmatch.fnmatch(path.name, pattern) for pattern in ALLOWED_ARTIFACT_PATTERNS)
    ]
    if unexpected:
        raise RuntimeError(f"soft-only artifact allowlist violation: {sorted(unexpected)}")


def _manifest(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    digest: str,
    semantics: Mapping[str, Any],
    profile: LossProfile,
) -> dict[str, Any]:
    if profile.legacy_terms:
        loss_weights = {
            "l2": profile.l2_weight,
            "l1": profile.l1_weight,
            "dice": 0.80,
            "ssim": 0.0 if args.disable_ssim else 0.18,
            "false_positive": 0.35,
            "gate": 0.0,
            "boundary": 0.0,
            "late_ssim": 0.0,
        }
    else:
        loss_weights = {
            "l2": profile.l2_weight,
            "l1": profile.l1_weight,
            "dice": 0.0,
            "ssim": 0.0,
            "false_positive": 0.0,
            "gate": args.lambda_g if profile.use_gate else 0.0,
            "boundary": args.lambda_b if profile.use_boundary else 0.0,
            "late_ssim": 0.0 if args.disable_ssim else args.late_ssim_weight,
        }
    return {
        "schema": "pcraft_soft_loss_v2_run_v1",
        "config_hash": digest,
        "config": config,
        "profile": profile.name,
        "resolved_view_aggregation": profile.view_aggregation,
        "loss_weights": loss_weights,
        "proxy": dict(semantics),
        "protocol": {
            "name": "sadekar_tfs_fixed_six_view",
            "camera_azimuth_deg": list(CAMERA_AZIMUTH_DEG),
            "camera_elevation_deg": list(CAMERA_ELEVATION_DEG),
            "projection_binary_rule": "per_view_prediction >= 0.8 * per_view_max",
            "density_thresholds": list(DENSITY_THRESHOLDS),
            "density_threshold_semantics": True,
        },
        "scope": {
            "soft_only": True,
            "final_step_export_only": True,
            "artifact_allowlist": list(ALLOWED_ARTIFACT_PATTERNS),
        },
        "determinism": {
            "seed": args.seed,
            "algorithms_requested": True,
            "warn_only": True,
            "cudnn_benchmark": False,
            "mode": "best_effort",
            "bitwise_reproducibility_guaranteed": False,
            "reason": "PyTorch3D/CUDA kernels may lack a strictly deterministic implementation.",
        },
    }


def run(args: argparse.Namespace, *, command_argv: Sequence[str] | None = None) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    resume_requested = args.resume is not None
    resume_payload: Mapping[str, Any] | None = None
    try:
        validate_args(args)
        if output.exists() and any(output.iterdir()) and not resume_requested:
            raise FileExistsError(
                f"refusing to overwrite non-empty run directory: {output}"
            )
        config = resolved_config(args)
        digest = config_hash(config)
        semantics = proxy_execution_semantics(args)
        profile = _profile_for_args(args)
        if resume_requested:
            checkpoint_path = (
                output / "checkpoint_latest.pt"
                if args.resume == "__OUTPUT_LATEST__"
                else Path(args.resume).resolve()
            )
            try:
                resume_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            except Exception as exc:
                raise ResumePreflightError(
                    f"could not load resume checkpoint {checkpoint_path}: {exc}"
                ) from exc
            _validate_checkpoint_metadata(resume_payload, config, digest)
    except ResumePreflightError:
        raise
    except Exception as exc:
        if resume_requested:
            raise ResumePreflightError(f"resume preflight failed: {exc}") from exc
        raise

    command = {
        "argv": list(command_argv if command_argv is not None else sys.argv),
        "resolved_config": config,
        "config_hash": digest,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "COMMAND.json", command)
    _atomic_json(output / "run_manifest.json", _manifest(args, config, digest, semantics, profile))
    _atomic_json(
        output / "STATUS.json",
        {"state": "running", "config_hash": digest, "last_completed_step": 0},
    )

    _seed_everything(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {args.device}")
    device = requested_device
    targets = load_targets(args.shadow_files, True, args.image_size, device)
    if targets.shape[0] != 6:
        raise RuntimeError("fixed protocol must produce exactly six targets")
    protocol = SadekarTFSProtocol(
        device=device,
        image_size=args.image_size,
        zdist=args.sadekar_zdist,
        volume_extent_world=args.sadekar_volume_extent,
        n_pts_per_ray=args.sadekar_npts_per_ray,
    )
    logits = torch.full(
        (1, 1, args.res, args.res, args.res),
        -4.0,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    start_step = 1
    last_record: dict[str, Any] | None = None
    if resume_requested:
        assert resume_payload is not None
        start_step, last_record = restore_checkpoint_payload(
            resume_payload,
            logits=logits,
            optimizer=optimizer,
            expected_config=config,
            expected_digest=digest,
        )

    cached_boundary_weights = boundary_weights(targets, sigma_px=2.0)
    history_path = output / "history.jsonl"
    for step in range(start_step, args.steps + 1):
        progress = step / float(args.steps)
        probability = torch.sigmoid(logits)
        raw_projection = protocol.render_all(probability)
        auxiliary_projection = training_auxiliary_projection(
            probability,
            raw_projection,
            args=args,
            protocol=protocol,
            progress=progress,
        )

        total, per_view_total, scalars = _compute_step_loss(
            raw_projection,
            auxiliary_projection,
            targets,
            profile=profile,
            args=args,
            progress=progress,
            cached_boundary_weights=cached_boundary_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([logits], max_norm=1.0)
        optimizer.step()
        last_record = {
            "step": step,
            "progress": progress,
            **scalars,
            "gradient_norm_pre_clip": float(gradient_norm.detach().cpu()),
            "per_view_loss": [
                float(value) for value in per_view_total.detach().cpu().tolist()
            ],
            "probability_mean": float(probability.detach().mean().cpu()),
            "probability_max": float(probability.detach().max().cpu()),
        }
        if step == 1 or step % args.save_every == 0 or step == args.steps:
            _append_history(history_path, last_record)
            checkpoint = build_checkpoint_payload(
                logits=logits,
                optimizer=optimizer,
                step=step,
                resolved_config=config,
                config_digest=digest,
                last_record=last_record,
            )
            _atomic_checkpoint(output / "checkpoint_latest.pt", checkpoint)
            _atomic_json(
                output / "STATUS.json",
                {
                    "state": "running",
                    "config_hash": digest,
                    "last_completed_step": step,
                },
            )
            print(json.dumps(last_record, sort_keys=True), flush=True)

    if not (output / "checkpoint_latest.pt").exists():
        checkpoint = build_checkpoint_payload(
            logits=logits,
            optimizer=optimizer,
            step=min(args.steps, start_step - 1),
            resolved_config=config,
            config_digest=digest,
            last_record=last_record,
        )
        _atomic_checkpoint(output / "checkpoint_latest.pt", checkpoint)
    _link_final_checkpoint(
        output / "checkpoint_latest.pt", output / "checkpoint_final.pt"
    )

    with torch.no_grad():
        probability = torch.sigmoid(logits)
        raw_projection = protocol.render_all(probability)
        final_proxy_projection: torch.Tensor | None = None
        if semantics["threshold_proxy_enabled"]:
            proxy_density = compact_soft_threshold(
                probability,
                tau=args.sadekar_density_threshold,
                margin=threshold_margin_schedule(1.0),
            )
            final_proxy_projection = protocol.render_all(proxy_density * 0.999)
        binary_volumes: dict[float, torch.Tensor] = {}
        binary_masks: dict[float, torch.Tensor] = {}
        for threshold in DENSITY_THRESHOLDS:
            volume = probability >= threshold
            binary_volumes[threshold] = volume
            binary_render = protocol.render_all(volume.to(probability.dtype) * 0.999)
            binary_masks[threshold] = protocol.binary_mask(binary_render)

    np.save(
        output / "occupancy_probability.npy",
        probability[0, 0].detach().cpu().numpy().astype(np.float32),
    )
    for threshold, volume in binary_volumes.items():
        label = _format_threshold(threshold)
        np.save(
            output / f"occupancy_binary_tau_{label}.npy",
            volume[0, 0].detach().cpu().numpy().astype(np.uint8),
        )
    for view_index in range(6):
        _write_image(output / f"gt_view_{view_index}.png", targets[view_index])
        _write_image(
            output / f"continuous_view_{view_index}.png", raw_projection[view_index]
        )
        if final_proxy_projection is not None:
            _write_image(
                output / f"proxy_view_{view_index}.png",
                final_proxy_projection[view_index],
            )
        for threshold, masks in binary_masks.items():
            label = _format_threshold(threshold)
            _write_image(
                output / f"binary_tau_{label}_view_{view_index}.png",
                masks[view_index],
                binary=True,
            )

    metric_report = evaluate_rendered_views(
        raw_projection,
        targets,
        binary_masks,
        proxy_prediction=final_proxy_projection,
        primary_threshold=PRIMARY_DENSITY_THRESHOLD,
        boundary_tolerance_px=2.0,
    )
    metric_report.update(
        {
            "schema": "pcraft_soft_loss_v2_metrics_v1",
            "config_hash": digest,
            "profile": args.profile,
            "steps": args.steps,
            "final_step_exported": True,
            "final_loss": None if last_record is None else last_record["loss"],
            "final_gradient_norm_pre_clip": (
                None if last_record is None else last_record["gradient_norm_pre_clip"]
            ),
        }
    )
    _atomic_json(output / "metrics.json", metric_report)
    _assert_artifact_allowlist(output)
    _atomic_json(
        output / "STATUS.json",
        {
            "state": "completed",
            "config_hash": digest,
            "last_completed_step": args.steps,
            "metrics_file": "metrics.json",
        },
    )
    return metric_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    try:
        run(parsed, command_argv=list(argv) if argv is not None else sys.argv)
    except ResumePreflightError:
        # The existing audit trail belongs to the original compatible run.
        # A rejected resume must not rewrite any of it, including STATUS.json.
        raise
    except Exception as exc:
        output = Path(parsed.output_dir).resolve()
        if output.exists() and output.is_dir():
            try:
                _atomic_json(
                    output / "STATUS.json",
                    {"state": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                )
            except Exception:
                pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
