"""
post_export.py
Training-end reporting, visualization, and paper-style figure export utilities.
"""
from __future__ import annotations

import json
import math
import os
import textwrap
import traceback
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from utils import get_tfs_view_specs

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

PHASE_COLORS = {
    "Phase 1": "#4C72B0",
    "Phase 2": "#DD8452",
    "Phase 3": "#55A868",
    "Phase 4": "#C44E52",
    "Phase 5": "#8172B2",
}

_MESH_CACHE: Dict[Tuple[str, str], Meshes] = {}
_RENDERER_CACHE: Dict[Tuple[str, int], MeshRenderer] = {}
_IMAGE_CACHE: Dict[Tuple[str, str, int, str, Tuple[Tuple[float, float], ...]], List[np.ndarray]] = {}


def _resolve_device(device_like: Any) -> torch.device:
    if isinstance(device_like, torch.device):
        return device_like
    if device_like is None:
        return torch.device("cpu")
    return torch.device(str(device_like))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_read_gray(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _safe_read_rgb(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    return img


def _resize_rgb(img: Optional[np.ndarray], size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    if img is None:
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "N/A", (width // 3, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas[..., ::-1]
    rgb = _ensure_rgb(img)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _sorted_view_files(run_dir: Path, pattern: str) -> List[str]:
    paths = list(glob(str(run_dir / pattern)))

    def _extract_idx(path_str: str) -> int:
        stem = Path(path_str).stem
        tail = stem.split("_")[-1]
        try:
            return int(tail)
        except ValueError:
            return 10 ** 9

    return [str(p) for p in sorted(paths, key=_extract_idx)]


def _extract_view_idx(path_str: str) -> Optional[int]:
    stem = Path(path_str).stem
    tail = stem.split("_")[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _filter_view_paths(paths: Sequence[str], expected_count: int) -> List[str]:
    if expected_count <= 0:
        return []
    indexed: Dict[int, str] = {}
    for path in paths:
        idx = _extract_view_idx(path)
        if idx is None or idx < 0 or idx >= expected_count:
            continue
        indexed.setdefault(idx, path)
    return [indexed[idx] for idx in range(expected_count) if idx in indexed]


def _first_existing(run_dir: Path, names: Sequence[str]) -> Optional[str]:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def _parse_obj_counts(obj_path: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not obj_path or not os.path.exists(obj_path):
        return None, None
    verts = 0
    faces = 0
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    verts += 1
                elif line.startswith("f "):
                    faces += 1
    except OSError:
        return None, None
    return verts, faces


def _seconds_to_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    total = int(round(float(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _metric_block(metrics_history: Dict[str, Any]) -> Dict[str, Any]:
    final = metrics_history.get("final_metrics", {})
    if "export_metrics" in final:
        return final["export_metrics"]
    return final


def _renderer_metric_block(metrics_history: Dict[str, Any]) -> Dict[str, Any]:
    final = metrics_history.get("final_metrics", {})
    return final.get("renderer_metrics", {})


def _estimate_flip_pair_score(view_paths: Sequence[str]) -> float:
    if len(view_paths) < 6:
        return 0.0
    scores = []
    for a_idx, b_idx in ((0, 3), (1, 4), (2, 5)):
        img_a = _safe_read_gray(view_paths[a_idx])
        img_b = _safe_read_gray(view_paths[b_idx])
        if img_a is None or img_b is None:
            continue
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)
        flip_a = np.fliplr(img_a)
        a_bin = flip_a > 127
        b_bin = img_b > 127
        union = np.logical_or(a_bin, b_bin).sum()
        score = 1.0 if union == 0 else float(np.logical_and(a_bin, b_bin).sum() / union)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def _infer_projection_mode(config: Dict[str, Any], artifact_paths: Dict[str, Any], view_count: int) -> str:
    if config:
        if config.get("projection_mode"):
            return str(config["projection_mode"])
        if config.get("tsf_mode"):
            return "tsf"
        if config.get("tsf4v_mode"):
            return "tsf4v"
        if config.get("mirror_mode") and config.get("num_total_views") == 6 and config.get("num_input_views") == 3:
            return "tsf" if config.get("camera_mode", "ortho") == "ortho" else "ring"

    candidate_paths = artifact_paths.get("gt_views") or artifact_paths.get("final_views") or []
    if view_count >= 6 and _estimate_flip_pair_score(candidate_paths) > 0.92:
        return "tsf"
    if view_count in (3, 4, 6, 8):
        return "ring"
    return "unknown"


def _detect_sample_id(run_dir: Path) -> int:
    patterns = [
        "sample_*_output.obj",
        "sample_*_metrics.txt",
        "sample_*_metrics_report.txt",
        "sample_*_view_0.png",
        "sample_*_pred_cleaned_view_0.png",
    ]
    for pattern in patterns:
        for path_str in glob(str(run_dir / pattern)):
            stem = Path(path_str).stem
            parts = stem.split("_")
            for idx, token in enumerate(parts):
                if token == "sample" and idx + 1 < len(parts):
                    try:
                        return int(parts[idx + 1])
                    except ValueError:
                        continue
    return 1


def _collect_artifact_paths(run_dir: Path, sample_id: int) -> Dict[str, Any]:
    obj_candidates = [
        f"sample_{sample_id}_output_stable_auto.obj",
        f"sample_{sample_id}_output_stable_mixed.obj",
        f"sample_{sample_id}_output_stable.obj",
        f"sample_{sample_id}_printready.obj",
        f"sample_{sample_id}_output.obj",
        "final.obj",
    ]
    rotation_candidates = [
        f"sample_{sample_id}_rotation.gif",
        f"sample_{sample_id}_output_stable_auto_rotation.gif",
        f"sample_{sample_id}_output_stable_mixed_rotation.gif",
    ]
    return {
        "obj": _first_existing(run_dir, obj_candidates),
        "voxels": _first_existing(run_dir, [f"sample_{sample_id}_final_voxels.npy"]),
        "metrics_history": str(run_dir / "metrics_history.json") if (run_dir / "metrics_history.json").exists() else None,
        "metrics_report": _first_existing(run_dir, [f"sample_{sample_id}_metrics_report.txt", "metrics_report.txt", f"sample_{sample_id}_metrics.txt"]),
        "rotation_gif": _first_existing(run_dir, rotation_candidates),
        "artistic_panel": _first_existing(run_dir, [f"sample_{sample_id}_projection_artistic.png"]),
        "artistic_panel_primary": _first_existing(run_dir, [f"sample_{sample_id}_projection_artistic_primary.png"]),
        "artistic_panel_opposite": _first_existing(run_dir, [f"sample_{sample_id}_projection_artistic_opposite.png"]),
        "sixview_panel": _first_existing(run_dir, [f"sample_{sample_id}_projection_sixview.png"]),
        "final_views": _sorted_view_files(run_dir, f"sample_{sample_id}_view_*.png"),
        "renderer_views": _sorted_view_files(run_dir, f"sample_{sample_id}_renderer_view_*.png"),
        "pred_cleaned_views": _sorted_view_files(run_dir, f"sample_{sample_id}_pred_cleaned_view_*.png"),
        "pred_raw_views": _sorted_view_files(run_dir, f"sample_{sample_id}_pred_raw_view_*.png"),
        "gt_views": _sorted_view_files(run_dir, "gt_view_*.png"),
        "comparison": _first_existing(run_dir, [
            f"sample_{sample_id}_output_stable_auto_silhouettes/comparison.png",
            f"sample_{sample_id}_output_stable_mixed_silhouettes/comparison.png",
            f"sample_{sample_id}_output_silhouettes/comparison.png",
        ]),
        "iou_summary": _first_existing(run_dir, [
            f"sample_{sample_id}_output_stable_auto_silhouettes/iou_summary.txt",
            f"sample_{sample_id}_output_stable_mixed_silhouettes/iou_summary.txt",
            f"sample_{sample_id}_output_silhouettes/iou_summary.txt",
        ]),
        "charts_dir": str(run_dir / "charts") if (run_dir / "charts").exists() else None,
        "paper_figures_dir": str(run_dir / "paper_figures") if (run_dir / "paper_figures").exists() else None,
    }


def _normalize_run_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    sample_id = _detect_sample_id(run_dir)
    artifact_paths = _collect_artifact_paths(run_dir, sample_id)
    metrics_path = artifact_paths.get("metrics_history")
    if not metrics_path:
        return None

    try:
        metrics_history = _read_json(Path(metrics_path))
    except Exception:
        return None

    final = metrics_history.get("final_metrics", {})
    export_metrics = _metric_block(metrics_history)
    renderer_metrics = _renderer_metric_block(metrics_history)
    config = final.get("config", {})
    geometry_type = "voxel" if "export_metrics" in final or run_dir.name.startswith("voxel") else "mesh"
    expected_view_count = (
        int(config.get("num_total_views", 0) or 0)
        or len(export_metrics.get("view_ious", []))
        or len(renderer_metrics.get("view_ious", []))
        or len(artifact_paths.get("final_views", []))
        or len(artifact_paths.get("gt_views", []))
    )
    if expected_view_count > 0:
        for key in ("final_views", "renderer_views", "pred_cleaned_views", "pred_raw_views", "gt_views"):
            artifact_paths[key] = _filter_view_paths(artifact_paths.get(key, []), expected_view_count)
    view_count = expected_view_count or len(artifact_paths.get("final_views", [])) or len(artifact_paths.get("gt_views", []))
    projection_mode = _infer_projection_mode(config, artifact_paths, view_count)
    flip_score = _estimate_flip_pair_score(artifact_paths.get("gt_views") or artifact_paths.get("final_views") or [])
    mirror_mode = bool(config.get("mirror_mode")) if "mirror_mode" in config else bool(flip_score > 0.92)
    base_views = config.get("num_input_views")
    if base_views is None:
        base_views = view_count // 2 if mirror_mode and view_count > 0 and view_count % 2 == 0 else view_count

    mean_iou = float(export_metrics.get("mean_iou", 0.0))
    artifact_score = int(bool(artifact_paths.get("obj"))) + int(bool(artifact_paths.get("voxels")))
    artifact_score += min(len(artifact_paths.get("final_views", [])), 6)
    artifact_score += min(len(artifact_paths.get("gt_views", [])), 6)

    return {
        "run_id": run_dir.name,
        "sample_id": sample_id,
        "source_dir": str(run_dir.resolve()),
        "source_project": "mesh_combined2" if "mesh_combined2" in str(run_dir.resolve()) else "mesh_combined",
        "geometry_type": geometry_type,
        "projection_mode": projection_mode,
        "camera_mode": config.get("camera_mode", "ortho"),
        "mirror_mode": mirror_mode,
        "export_strategy": final.get("export_strategy", "mesh_output"),
        "aggregate_metrics": {
            "mean_iou": mean_iou,
            "mean_dice": float(export_metrics.get("mean_dice", 0.0)),
            "normal_consistency": final.get("normal_consistency"),
            "occupied_voxels": export_metrics.get("occupied_voxels"),
        },
        "renderer_metrics": renderer_metrics,
        "per_view_metrics": {
            "view_ious": [float(v) for v in export_metrics.get("view_ious", [])],
            "view_dices": [float(v) for v in export_metrics.get("view_dices", [])],
        },
        "phase_metrics": final.get("phase_best_ious", {}),
        "artifact_paths": artifact_paths,
        "config": config,
        "history": metrics_history.get("history", []),
        "view_count": view_count,
        "base_view_count": int(base_views) if base_views is not None else view_count,
        "flip_pair_score": float(flip_score),
        "quality_score": mean_iou + artifact_score * 1e-3,
        "metrics_history": metrics_history,
    }


def _default_history_roots(project_root: Path) -> List[Path]:
    roots = [project_root / "output", project_root.parent / "mesh_combined" / "output"]
    result = []
    for root in roots:
        if root.exists():
            result.append(root)
    return result


def scan_history_runs(current_output_dir: str, history_roots: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    project_root = Path(__file__).resolve().parent
    roots = [Path(p).resolve() for p in history_roots] if history_roots else _default_history_roots(project_root)
    current_output = Path(current_output_dir).resolve()
    if current_output.parent.exists():
        roots = [current_output.parent] + [root for root in roots if root != current_output.parent]

    seen = set()
    summaries: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*")):
            if not run_dir.is_dir():
                continue
            resolved = str(run_dir.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            summary = _normalize_run_summary(run_dir)
            if summary is not None:
                summaries.append(summary)
    summaries.sort(key=lambda x: x["quality_score"], reverse=True)
    return summaries


def _pick_best_summary(
    summaries: Sequence[Dict[str, Any]],
    geometry_type: Optional[str] = None,
    projection_mode: Optional[str] = None,
    base_view_count: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    filtered = []
    for summary in summaries:
        if geometry_type and summary["geometry_type"] != geometry_type:
            continue
        if projection_mode and summary["projection_mode"] != projection_mode:
            continue
        if base_view_count is not None and summary["base_view_count"] != base_view_count:
            continue
        filtered.append(summary)
    if not filtered:
        return None
    filtered.sort(key=lambda x: x["quality_score"], reverse=True)
    return filtered[0]


def _phase_order_key(phase_name: str) -> int:
    try:
        return int(str(phase_name).split()[-1])
    except Exception:
        return 10 ** 9


def plot_convergence_curves(metrics_history: Dict[str, Any], charts_dir: Path) -> Optional[str]:
    history = metrics_history.get("history", [])
    if not history:
        return None

    iters = [h.get("global_iter", idx) for idx, h in enumerate(history)]
    phases = [h.get("phase", "Phase 1") for h in history]
    use_normalized_loss = any(h.get("normalized_total_loss") is not None for h in history)
    losses = [h.get("normalized_total_loss", h.get("total_loss")) for h in history]
    unique_phases = sorted(set(phases), key=_phase_order_key)
    relative_losses: List[Optional[float]] = [None] * len(losses)
    if len(unique_phases) > 1:
        for pname in unique_phases:
            phase_indices = [i for i, phase_name in enumerate(phases) if phase_name == pname and losses[i] is not None]
            if not phase_indices:
                continue
            start_val = float(losses[phase_indices[0]])
            denom = max(abs(start_val), 1e-8)
            for i in phase_indices:
                relative_losses[i] = float(losses[i]) / denom
    else:
        relative_losses = [None if v is None else float(v) for v in losses]
    ious = [h.get("mean_iou") for h in history]
    dices = [h.get("mean_dice") for h in history]
    ncs = [h.get("normal_consistency") for h in history]
    final_nc = metrics_history.get("final_metrics", {}).get("normal_consistency")
    if not any(v is not None for v in ncs) and final_nc is not None and ncs:
        ncs[-1] = float(final_nc)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Training Convergence Curves", fontsize=18, fontweight="bold", y=0.98)

    for ax, values, title, ylabel in (
        (
            axes[0, 0],
            relative_losses,
            "Loss Relative to Phase Start" if len(unique_phases) > 1 else ("Normalized Total Loss" if use_normalized_loss else "Total Loss"),
            "Loss / Phase Start" if len(unique_phases) > 1 else ("Normalized Loss" if use_normalized_loss else "Loss"),
        ),
        (axes[0, 1], ious, "Mean IoU", "IoU"),
        (axes[1, 0], dices, "Mean Dice", "Dice"),
        (axes[1, 1], ncs, "Normal Consistency Loss", "Loss (lower is better)"),
    ):
        numeric_values = [None if v is None else float(v) for v in values]
        if not any(v is not None for v in numeric_values):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=13)
            ax.set_title(title)
            continue

        if title == "Normal Consistency Loss":
            valid_points = [(iters[i], numeric_values[i], phases[i]) for i in range(len(iters)) if numeric_values[i] is not None]
            if len(valid_points) <= 2:
                for x, y, pname in valid_points:
                    color = PHASE_COLORS.get(pname, "#777777")
                    ax.scatter([x], [y], color=color, s=52, zorder=3, label=pname)
                ax.set_title("Normal Consistency Loss (Sampled Checkpoints)")
                ax.set_xlabel("Global Iteration")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                if valid_points:
                    ax.text(0.5, 0.92, "Sparse NC samples in this run", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="#666666")
                if len(set(p for _, _, p in valid_points)) > 1:
                    ax.legend(framealpha=0.85)
                continue

        for pname in unique_phases:
            color = PHASE_COLORS.get(pname, "#777777")
            xs = [iters[i] for i in range(len(iters)) if phases[i] == pname and numeric_values[i] is not None]
            ys = [numeric_values[i] for i in range(len(numeric_values)) if phases[i] == pname and numeric_values[i] is not None]
            if xs:
                ax.plot(xs, ys, color=color, linewidth=1.6, label=pname, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if title in ("Mean IoU", "Mean Dice"):
            ax.set_ylim(0, 1.05)
        if len(unique_phases) > 1:
            ax.legend(framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = charts_dir / "convergence_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def plot_loss_components(metrics_history: Dict[str, Any], charts_dir: Path) -> Optional[str]:
    history = metrics_history.get("history", [])
    if not history or not any("loss_components" in h for h in history):
        return None

    keys = set()
    for h in history:
        keys.update(h.get("loss_components", {}).keys())
    if not keys:
        return None

    priority = [
        "silhouette", "l1", "ms_ssim", "iou", "dice", "asymmetric",
        "false_negative", "focal_tversky", "boundary", "reg_tv", "reg_sparsity",
        "isolation", "projection_l1", "projection_iou", "projection_tversky",
        "projection_boundary",
    ]
    ordered = [k for k in priority if k in keys] + [k for k in sorted(keys) if k not in priority]
    ncols = 3
    nrows = math.ceil(len(ordered) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.8 * nrows))
    axes = np.atleast_2d(axes)
    fig.suptitle("Loss Components Over Training", fontsize=18, fontweight="bold", y=0.995)

    iters = [h.get("global_iter", idx) for idx, h in enumerate(history)]
    title_map = {
        "ms_ssim": "MS-SSIM",
        "iou": "IoU Loss",
        "dice": "Dice Loss",
        "l1": "L1",
        "reg_tv": "TV Regularization",
        "reg_sparsity": "Sparsity Regularization",
        "projection_l1": "Projection L1",
        "projection_iou": "Projection IoU",
        "projection_tversky": "Projection Tversky",
        "projection_boundary": "Projection Boundary",
    }
    for idx, key in enumerate(ordered):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        values = [h.get("loss_components", {}).get(key) for h in history]
        if any(v is not None for v in values):
            ax.plot(iters, [np.nan if v is None else float(v) for v in values], color=plt.cm.tab20(idx % 20), linewidth=1.4)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(title_map.get(key, key.replace("_", " ").title()))
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)

    for idx in range(len(ordered), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out_path = charts_dir / "loss_components.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def plot_quantitative_metrics(metrics_history: Dict[str, Any], charts_dir: Path) -> Optional[str]:
    final = metrics_history.get("final_metrics", {})
    metric_block = _metric_block(metrics_history)
    view_ious = metric_block.get("view_ious", [])
    view_dices = metric_block.get("view_dices", [])
    if not view_ious:
        return None

    mean_iou = float(metric_block.get("mean_iou", 0.0))
    mean_dice = float(metric_block.get("mean_dice", 0.0))
    normal_consistency = final.get("normal_consistency")

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    x = np.arange(len(view_ious))
    ax1.bar(x, view_ious, color="#4C72B0", alpha=0.9, edgecolor="white")
    ax1.axhline(mean_iou, color="#C44E52", linestyle="--", linewidth=2, label=f"Mean={mean_iou:.4f}")
    ax1.set_title("IoU per View")
    ax1.set_xlabel("View Index")
    ax1.set_ylabel("IoU")
    ax1.set_ylim(max(0.0, min(view_ious) - 0.06), 1.01)
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.legend(loc="lower right")

    ax2.bar(x, view_dices, color="#55A868", alpha=0.9, edgecolor="white")
    ax2.axhline(mean_dice, color="#C44E52", linestyle="--", linewidth=2, label=f"Mean={mean_dice:.4f}")
    ax2.set_title("Dice per View")
    ax2.set_xlabel("View Index")
    ax2.set_ylabel("Dice")
    ax2.set_ylim(max(0.0, min(view_dices) - 0.06), 1.01)
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend(loc="lower right")

    ax3.axis("off")
    col_labels = [f"V{i}" for i in range(len(view_ious))] + ["Mean", "Normal Consistency\n(loss)"]
    table_data = [
        [f"{v:.4f}" for v in view_ious] + [f"{mean_iou:.4f}", "N/A" if normal_consistency is None else f"{float(normal_consistency):.4f}"],
        [f"{v:.4f}" for v in view_dices] + [f"{mean_dice:.4f}", "—"],
    ]
    table = ax3.table(cellText=table_data, rowLabels=["IoU", "Dice"], colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#F7F7F7" if row % 2 == 0 else "#FFFFFF")
    ax3.set_title("Quantitative Metrics Summary", fontsize=14, fontweight="bold", pad=14)

    out_path = charts_dir / "quantitative_metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def plot_inconsistency_maps(output_dir: Path, charts_dir: Path, sample_id: int) -> Optional[str]:
    gt_files = _sorted_view_files(output_dir, "gt_view_*.png")
    pred_files = _sorted_view_files(output_dir, f"sample_{sample_id}_pred_cleaned_view_*.png")
    if not pred_files:
        pred_files = _sorted_view_files(output_dir, f"sample_{sample_id}_view_*.png")
    if not pred_files:
        pred_files = _sorted_view_files(output_dir, f"sample_{sample_id}_renderer_view_*.png")
    if not gt_files or not pred_files:
        return None

    n_views = min(len(gt_files), len(pred_files), 6)
    ncols = min(3, n_views)
    nrows = int(math.ceil(n_views / ncols)) * 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    fig.suptitle("Shadow Inconsistency Maps", fontsize=18, fontweight="bold", y=0.995)

    cmap = ListedColormap([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.26, 0.52, 0.96, 1.0],
        [1.0, 0.60, 0.20, 1.0],
    ])

    for idx in range(n_views):
        row_offset = (idx // ncols) * 3
        col = idx % ncols
        gt = _safe_read_gray(gt_files[idx])
        pred = _safe_read_gray(pred_files[idx])
        if gt is None or pred is None:
            continue
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_bool = gt > 127
        pred_bool = pred > 127
        diff = np.zeros_like(gt, dtype=np.uint8)
        diff[gt_bool & pred_bool] = 1
        diff[~gt_bool & pred_bool] = 2
        diff[gt_bool & ~pred_bool] = 3

        axes[row_offset + 0, col].imshow(gt, cmap="gray")
        axes[row_offset + 0, col].set_title(f"View {idx} Target")
        axes[row_offset + 1, col].imshow(pred, cmap="gray")
        axes[row_offset + 1, col].set_title(f"View {idx} Predicted")
        axes[row_offset + 2, col].imshow(diff, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        fp = int((diff == 2).sum())
        fn = int((diff == 3).sum())
        axes[row_offset + 2, col].set_title(f"View {idx} Diff\nFP={fp} FN={fn}")
        for ax in (axes[row_offset + 0, col], axes[row_offset + 1, col], axes[row_offset + 2, col]):
            ax.axis("off")

    out_path = charts_dir / "inconsistency_maps.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def plot_phase_comparison(metrics_history: Dict[str, Any], charts_dir: Path) -> Optional[str]:
    phase_ious = metrics_history.get("final_metrics", {}).get("phase_best_ious", {})
    if not phase_ious:
        return None

    phases = list(phase_ious.keys())
    values = [float(v) for v in phase_ious.values()]
    colors = [PHASE_COLORS.get(p, "#777777") for p in phases]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bars = ax.bar(phases, values, color=colors, edgecolor="white", linewidth=1.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.003, f"{val:.4f}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Best IoU at Each Phase", fontsize=16, fontweight="bold")
    ax.set_ylabel("Best IoU")
    ax.set_ylim(max(0.0, min(values) - 0.05), 1.01)
    ax.grid(True, axis="y", alpha=0.2)
    out_path = charts_dir / "phase_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def plot_radar_chart(metrics_history: Dict[str, Any], charts_dir: Path) -> Optional[str]:
    metric_block = _metric_block(metrics_history)
    view_ious = metric_block.get("view_ious", [])
    view_dices = metric_block.get("view_dices", [])
    if not view_ious or not view_dices:
        return None

    final = metrics_history.get("final_metrics", {})
    labels = ["Mean IoU", "Mean Dice", "Min IoU", "Min Dice"]
    values = [
        float(metric_block.get("mean_iou", 0.0)),
        float(metric_block.get("mean_dice", 0.0)),
        float(min(view_ious)),
        float(min(view_dices)),
    ]
    normal_consistency = final.get("normal_consistency")
    if normal_consistency is not None:
        nc_quality = 1.0 / (1.0 + float(normal_consistency))
        labels.insert(2, "NC\nQuality")
        values.insert(2, float(nc_quality))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.fill(angles_closed, values_closed, color="#4C72B0", alpha=0.25)
    ax.plot(angles_closed, values_closed, "o-", color="#4C72B0", linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Overall Quality Radar Chart", fontsize=16, fontweight="bold", pad=24)
    for angle, val in zip(angles, values):
        ax.text(angle, min(val + 0.06, 1.05), f"{val:.4f}", ha="center", va="center", fontsize=9)
    if normal_consistency is None:
        fig.text(0.5, 0.06, "Normal Consistency: N/A for current voxel pipeline", ha="center", va="center", fontsize=10, color="#666666")
    else:
        fig.text(0.5, 0.06, f"Raw normal consistency loss: {float(normal_consistency):.4f} (lower is better)", ha="center", va="center", fontsize=10, color="#666666")

    out_path = charts_dir / "radar_chart.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def generate_analytic_charts(metrics_history: Dict[str, Any], output_dir: str, sample_id: int) -> Dict[str, Optional[str]]:
    charts_dir = Path(output_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = {
        "convergence_curves": plot_convergence_curves(metrics_history, charts_dir),
        "loss_components": plot_loss_components(metrics_history, charts_dir),
        "quantitative_metrics": plot_quantitative_metrics(metrics_history, charts_dir),
        "inconsistency_maps": plot_inconsistency_maps(Path(output_dir), charts_dir, sample_id),
        "phase_comparison": plot_phase_comparison(metrics_history, charts_dir),
        "radar_chart": plot_radar_chart(metrics_history, charts_dir),
    }
    _json_dump(charts_dir / "index.json", chart_paths)
    return chart_paths


def _backfill_normal_consistency(summary: Dict[str, Any], _device: torch.device) -> None:
    final = summary["metrics_history"].setdefault("final_metrics", {})
    normal_consistency = final.get("normal_consistency")
    if normal_consistency is not None:
        summary["aggregate_metrics"]["normal_consistency"] = float(normal_consistency)


def _backfill_normalized_losses(summary: Dict[str, Any]) -> None:
    history = summary["metrics_history"].get("history", [])
    if not history or all(h.get("normalized_total_loss") is not None for h in history):
        return

    updated = False
    for entry in history:
        if entry.get("normalized_total_loss") is not None:
            continue
        total_loss = entry.get("total_loss")
        components = entry.get("loss_components") or {}
        if total_loss is None or not components:
            continue

        phase = str(entry.get("phase", ""))
        weight_sum = None

        if phase == "Phase 1":
            weight_sum = 7.0 + 6.0 + 2.5 + 0.3 + 0.02
            if float(components.get("iou", 0.0)) > 0.0 or float(components.get("dice", 0.0)) > 0.0:
                weight_sum += 3.0 + 2.0
            if any(float(components.get(key, 0.0)) > 0.0 for key in ("projection_l1", "projection_iou", "projection_tversky")):
                weight_sum += 2.0 + 1.5 + 1.0
        elif phase == "Phase 2":
            known = (
                float(components.get("asymmetric", 0.0)) * 8.0 +
                float(components.get("l1", 0.0)) * 6.0 +
                float(components.get("ms_ssim", 0.0)) * 3.0 +
                float(components.get("iou", 0.0)) * 5.0 +
                float(components.get("dice", 0.0)) * 3.0 +
                float(components.get("reg_tv", 0.0)) * 0.15 +
                float(components.get("reg_sparsity", 0.0)) * 0.03
            )
            if any(float(components.get(key, 0.0)) > 0.0 for key in ("projection_l1", "projection_iou", "projection_tversky", "projection_boundary")):
                known += (
                    float(components.get("projection_l1", 0.0)) * 3.0 +
                    float(components.get("projection_iou", 0.0)) * 2.0 +
                    float(components.get("projection_tversky", 0.0)) * 1.5 +
                    float(components.get("projection_boundary", 0.0)) * 0.8
                )
                weight_sum = 8.0 + 6.0 + 3.0 + 5.0 + 3.0 + 0.15 + 0.03 + 3.0 + 2.0 + 1.5 + 0.8
            else:
                weight_sum = 8.0 + 6.0 + 3.0 + 5.0 + 3.0 + 0.15 + 0.03

            isolation_val = float(components.get("isolation", 0.0))
            if abs(isolation_val) > 1e-8:
                inferred_iso_w = max(0.0, min(0.2, (float(total_loss) - known) / isolation_val))
                weight_sum += inferred_iso_w
            else:
                weight_sum += 0.05
        elif phase == "Phase 3":
            weight_sum = 7.0 + 3.0 + 5.0 + 7.0 + 3.0 + 0.1 + 0.04
            if any(float(components.get(key, 0.0)) > 0.0 for key in ("projection_l1", "projection_iou", "projection_tversky", "projection_boundary")):
                weight_sum += 3.5 + 2.5 + 2.0 + 1.2

        if weight_sum is None or weight_sum <= 0:
            continue
        entry["normalized_total_loss"] = float(total_loss) / float(weight_sum)
        updated = True

    if updated:
        metrics_history_path = summary["artifact_paths"].get("metrics_history")
        if metrics_history_path:
            _json_dump(Path(metrics_history_path), summary["metrics_history"])


def _get_mesh_renderer(device: torch.device, image_size: int) -> MeshRenderer:
    key = (str(device), int(image_size))
    if key in _RENDERER_CACHE:
        return _RENDERER_CACHE[key]

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(
        device=device,
        location=[[2.5, 2.2, -3.0]],
        ambient_color=((0.65, 0.65, 0.65),),
        diffuse_color=((0.55, 0.55, 0.55),),
        specular_color=((0.18, 0.18, 0.18),),
    )
    shader = HardPhongShader(
        device=device,
        lights=lights,
        blend_params=BlendParams(background_color=(0.98, 0.98, 0.99)),
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=shader,
    )
    _RENDERER_CACHE[key] = renderer
    return renderer


def _build_mesh_from_verts_faces(verts: np.ndarray, faces: np.ndarray, device: torch.device,
                                 color: Tuple[float, float, float] = (0.72, 0.25, 0.18)) -> Meshes:
    verts_t = torch.from_numpy(np.asarray(verts, dtype=np.float32)).to(device)
    faces_t = torch.from_numpy(np.asarray(faces, dtype=np.int64)).to(device)
    mins = verts_t.min(dim=0).values
    maxs = verts_t.max(dim=0).values
    center = (mins + maxs) * 0.5
    scale = (maxs - mins).max().clamp(min=1e-6)
    verts_t = (verts_t - center) / scale * 1.65
    verts_features = torch.ones((1, verts_t.shape[0], 3), device=device) * torch.tensor(color, device=device).view(1, 1, 3)
    textures = TexturesVertex(verts_features=verts_features)
    return Meshes(verts=[verts_t], faces=[faces_t], textures=textures)


def _mesh_from_binary_volume(volume: np.ndarray, device: torch.device) -> Optional[Meshes]:
    if volume.ndim == 4:
        volume = volume[0]
    volume = volume.astype(np.float32)
    if volume.max() <= 0:
        return None
    max_res = 160
    if max(volume.shape) > max_res:
        scale = max_res / float(max(volume.shape))
        volume = zoom(volume, zoom=(scale, scale, scale), order=0)
    padded = np.pad(volume, 1, mode="constant")
    padded = gaussian_filter(padded, sigma=0.65)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
    except Exception:
        return None
    verts = verts - 1.0
    verts = verts / (np.array(volume.shape, dtype=np.float32) - 1.0 + 1e-6) * 1.7 - 0.85
    return _build_mesh_from_verts_faces(verts, faces, device)


def _mesh_from_obj(obj_path: str, device: torch.device) -> Optional[Meshes]:
    if not obj_path or not os.path.exists(obj_path):
        return None
    try:
        verts, faces, _ = load_obj(obj_path, load_textures=False, device=device)
    except Exception:
        return None
    return _build_mesh_from_verts_faces(verts.detach().cpu().numpy(), faces.verts_idx.detach().cpu().numpy(), device)


def _load_display_mesh(summary: Dict[str, Any], device: torch.device) -> Optional[Meshes]:
    cache_key = (summary["source_dir"], str(device))
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key]

    mesh = None
    voxel_path = summary["artifact_paths"].get("voxels")
    if summary["geometry_type"] == "voxel" and voxel_path and os.path.exists(voxel_path):
        try:
            volume = np.load(voxel_path)
            mesh = _mesh_from_binary_volume(volume, device)
        except Exception:
            mesh = None
    if mesh is None:
        mesh = _mesh_from_obj(summary["artifact_paths"].get("obj"), device)
    if mesh is not None:
        _MESH_CACHE[cache_key] = mesh
    return mesh


def _get_view_angles(summary: Dict[str, Any], view_limit: Optional[int] = None) -> List[Tuple[float, float]]:
    num_views = int(summary.get("view_count") or len(summary.get("per_view_metrics", {}).get("view_ious", [])) or 0)
    if view_limit is not None:
        num_views = min(num_views, view_limit)
    projection_mode = summary.get("projection_mode", "unknown")
    if projection_mode == "tsf":
        config = summary.get("config", {})
        num_input_views = int(config.get("num_input_views", summary.get("base_view_count", num_views)) or 0)
        mirror_mode = bool(summary.get("mirror_mode"))
        specs = get_tfs_view_specs(num_input_views, mirror_mode)
        return [(float(elev), float(azim)) for _, elev, azim in specs[:num_views]]
    if projection_mode == "tsf4v":
        base = [(0.0, 0.0), (0.0, 60.0), (0.0, 120.0), (90.0, 0.0), (0.0, 180.0), (0.0, 240.0), (0.0, 300.0), (270.0, 0.0)]
        return base[:num_views]

    if num_views <= 0:
        return []
    mirror_mode = bool(summary.get("mirror_mode"))
    if mirror_mode:
        azims = np.linspace(0.0, 360.0 - 360.0 / num_views, num_views)
    else:
        azims = np.linspace(0.0, 180.0 - 180.0 / max(num_views, 1), num_views)
    return [(0.0, float(az)) for az in azims]


def _get_view_labels(summary: Dict[str, Any], view_limit: Optional[int] = None) -> List[str]:
    if summary.get("projection_mode") == "tsf":
        config = summary.get("config", {})
        num_views = int(summary.get("view_count") or len(summary.get("per_view_metrics", {}).get("view_ious", [])) or 0)
        if view_limit is not None:
            num_views = min(num_views, view_limit)
        num_input_views = int(config.get("num_input_views", summary.get("base_view_count", num_views)) or 0)
        mirror_mode = bool(summary.get("mirror_mode"))
        specs = get_tfs_view_specs(num_input_views, mirror_mode)
        return [label for label, _, _ in specs[:num_views]]
    angles = _get_view_angles(summary, view_limit=view_limit)
    if summary.get("projection_mode") == "tsf4v":
        names = ["v0", "v1", "v2", "top", "v3", "v4", "v5", "down"]
        return names[:len(angles)]
    return [f"az {int(round(az))} / el {int(round(el))}" for el, az in angles]


def _apply_d4_transform_gray(img: np.ndarray, op_id: int) -> np.ndarray:
    rot_k = int(op_id) % 4
    out = np.rot90(img, rot_k)
    if int(op_id) >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _apply_d4_transform_rgb(img: np.ndarray, op_id: int) -> np.ndarray:
    rot_k = int(op_id) % 4
    out = np.rot90(img, rot_k, axes=(0, 1))
    if int(op_id) >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 127
    b = mask_b > 127
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _render_mesh_rgb_and_masks(mesh: Meshes, angles: Sequence[Tuple[float, float]], device: torch.device,
                               image_size: int = 320, camera_mode: str = "ortho", dist: float = 2.55) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if mesh is None or not angles:
        return [], []
    renderer = _get_mesh_renderer(device, image_size)
    with torch.no_grad():
        elevs = torch.tensor([float(elev) for elev, _ in angles], device=device)
        azims = torch.tensor([float(azim) for _, azim in angles], device=device)
        R, T = look_at_view_transform(dist=dist, elev=elevs, azim=azims, device=device)
        if camera_mode == "perp":
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        else:
            cameras = FoVOrthographicCameras(device=device, R=R, T=T)
        batched_mesh = mesh.extend(len(angles))
        rgbs = renderer(meshes_world=batched_mesh, cameras=cameras)[..., :3].clamp(0.0, 1.0)
        fragments = renderer.rasterizer(meshes_world=batched_mesh, cameras=cameras)
        masks = (fragments.pix_to_face[..., 0] >= 0).cpu().numpy().astype(np.uint8) * 255
    rgb_list = [(img.cpu().numpy() * 255).astype(np.uint8) for img in rgbs]
    mask_list = [np.ascontiguousarray(mask) for mask in masks]
    return rgb_list, mask_list


def _render_mesh_views(mesh: Meshes, angles: Sequence[Tuple[float, float]], device: torch.device,
                       image_size: int = 320, camera_mode: str = "ortho", dist: float = 2.55) -> List[np.ndarray]:
    rgbs, _ = _render_mesh_rgb_and_masks(
        mesh,
        angles,
        device=device,
        image_size=image_size,
        camera_mode=camera_mode,
        dist=dist,
    )
    return rgbs


def _render_summary_views(summary: Dict[str, Any], angles: Sequence[Tuple[float, float]], device: torch.device,
                          image_size: int = 320, camera_mode: Optional[str] = None, dist: float = 2.55) -> List[np.ndarray]:
    rounded_angles = tuple((round(float(elev), 2), round(float(azim), 2)) for elev, azim in angles)
    mode = camera_mode or summary.get("camera_mode", "ortho")
    cache_key = (summary["source_dir"], str(device), int(image_size), str(mode), rounded_angles)
    if cache_key in _IMAGE_CACHE:
        return _IMAGE_CACHE[cache_key]
    mesh = _load_display_mesh(summary, device)
    imgs = _render_mesh_views(mesh, angles, device=device, image_size=image_size, camera_mode=mode, dist=dist)
    _IMAGE_CACHE[cache_key] = imgs
    return imgs


def _render_summary_views_aligned_to_targets(summary: Dict[str, Any], target_view_paths: Sequence[str], device: torch.device,
                                             image_size: int = 240) -> List[np.ndarray]:
    if summary.get("projection_mode") != "tsf":
        return []
    if not target_view_paths:
        return []

    mesh = _load_display_mesh(summary, device)
    if mesh is None:
        return []

    view_limit = min(len(target_view_paths), int(summary.get("view_count") or len(target_view_paths) or 0))
    candidate_angles = _get_view_angles(summary, view_limit=view_limit)
    if not candidate_angles:
        return []
    rgb_candidates, mask_candidates = _render_mesh_rgb_and_masks(
        mesh,
        candidate_angles,
        device=device,
        image_size=image_size,
        camera_mode=summary.get("camera_mode", "ortho"),
    )
    if not rgb_candidates:
        return []

    aligned_views: List[np.ndarray] = []
    for path in target_view_paths[:view_limit]:
        target = _safe_read_gray(path)
        if target is None:
            aligned_views.append(np.full((image_size, image_size, 3), 245, dtype=np.uint8))
            continue
        target = cv2.resize(target, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        best_score = -1.0
        best_rgb = None
        for cand_rgb, cand_mask in zip(rgb_candidates, mask_candidates):
            for op_id in range(8):
                transformed_mask = _apply_d4_transform_gray(cand_mask, op_id)
                score = _binary_iou(transformed_mask, target)
                if score > best_score:
                    best_score = score
                    best_rgb = _apply_d4_transform_rgb(cand_rgb, op_id)
        aligned_views.append(best_rgb if best_rgb is not None else np.full((image_size, image_size, 3), 245, dtype=np.uint8))
    return aligned_views


def _render_turntable(mesh: Meshes, device: torch.device, frames: int, fps: int,
                      camera_mode: str = "ortho", out_path: Optional[Path] = None) -> Optional[str]:
    if mesh is None:
        return None
    azims = np.linspace(0.0, 360.0, frames, endpoint=False)
    imgs = _render_mesh_views(mesh, [(20.0, float(az)) for az in azims], device, image_size=180, camera_mode=camera_mode, dist=2.55)
    if not imgs:
        return None
    if out_path is None:
        return None
    _ensure_parent(out_path)
    imageio.mimsave(out_path, imgs, duration=1.0 / max(fps, 1), loop=0)
    return str(out_path)


def _overlay_foreground_bgr(canvas_bgr: np.ndarray, fg_rgb: np.ndarray, top_left: Tuple[int, int]) -> None:
    fg_bgr = fg_rgb[..., ::-1]
    x0, y0 = top_left
    h, w = fg_bgr.shape[:2]
    x1 = min(canvas_bgr.shape[1], x0 + w)
    y1 = min(canvas_bgr.shape[0], y0 + h)
    if x0 >= x1 or y0 >= y1:
        return
    patch = fg_bgr[:y1 - y0, :x1 - x0]
    bg = canvas_bgr[y0:y1, x0:x1]
    mask = np.mean(patch, axis=2) < 245
    bg[mask] = patch[mask]


def _warp_mask_to_quad(mask: np.ndarray, quad: Sequence[Tuple[int, int]], canvas_size: Tuple[int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = np.array(quad, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(mask, M, canvas_size, flags=cv2.INTER_LINEAR)


def _projection_plane_specs(
    projection_mode: str,
    view_count: int,
    silhouette_labels: Optional[Sequence[str]] = None,
    view_angles: Optional[Sequence[Tuple[float, float]]] = None,
    base_view_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    labels = list(silhouette_labels or [])

    def _ring_like_specs() -> List[Dict[str, Any]]:
        palette = [
            (234, 220, 243),
            (229, 206, 236),
            (220, 240, 240),
            (242, 233, 206),
            (214, 231, 248),
            (245, 227, 237),
            (229, 238, 210),
            (218, 221, 245),
        ]
        specs: List[Dict[str, Any]] = []
        center_x = 590.0
        center_y = 360.0
        radius_x = 420.0
        for idx, (_, azim) in enumerate((view_angles or [])[:view_count]):
            theta = np.deg2rad(float(azim) % 360.0)
            x = center_x + radius_x * np.cos(theta)
            depth = np.sin(theta)
            width = 88.0 + 54.0 * (depth + 1.0) * 0.5
            height = 188.0 + 110.0 * (depth + 1.0) * 0.5
            shear = np.sign(np.cos(theta) if abs(np.cos(theta)) > 1e-4 else 1.0) * (16.0 + 18.0 * abs(np.cos(theta)))
            top = center_y - height * 0.52 - 34.0 * depth
            bottom = top + height
            left = x - width * 0.5
            right = x + width * 0.5
            quad = [
                (int(round(left + shear)), int(round(top))),
                (int(round(right + shear)), int(round(top + 16.0))),
                (int(round(right - shear)), int(round(bottom))),
                (int(round(left - shear)), int(round(bottom - 16.0))),
            ]
            label_y = min(750, int(round(bottom + 34.0)))
            label_x = int(round(x - 68.0))
            specs.append(
                {
                    "quad": quad,
                    "color": palette[idx % len(palette)],
                    "label_pos": (label_x, label_y),
                    "layer": 0 if depth < -0.15 else (1 if depth < 0.25 else 2),
                }
            )
        order = sorted(range(len(specs)), key=lambda idx: specs[idx]["label_pos"][0])
        last_x = -10**9
        for rank, spec_idx in enumerate(order):
            x, y = specs[spec_idx]["label_pos"]
            min_x = last_x + 102
            if x < min_x:
                x = min_x
                y = min(748, y + (18 if rank % 2 else 0))
            specs[spec_idx]["label_pos"] = (min(1035, int(x)), int(y))
            last_x = specs[spec_idx]["label_pos"][0]
        return specs

    if projection_mode == "tsf" and view_count >= 6:
        if view_count > 6 and view_angles:
            return _ring_like_specs()
        return [
            {
                "quad": [(432, 264), (566, 264), (566, 520), (432, 520)],
                "color": (245, 240, 245),
                "label_pos": (570, 546),
                "layer": 3,
            },
            {
                "quad": [(792, 146), (998, 202), (998, 568), (792, 514)],
                "color": (220, 204, 236),
                "label_pos": (860, 604),
                "layer": 1,
            },
            {
                "quad": [(292, 88), (688, 88), (854, 150), (126, 150)],
                "color": (238, 233, 203),
                "label_pos": (488, 180),
                "layer": 0,
            },
            {
                "quad": [(354, 198), (642, 198), (642, 544), (354, 544)],
                "color": (243, 236, 231),
                "label_pos": (648, 226),
                "layer": 0,
            },
            {
                "quad": [(86, 202), (292, 146), (292, 514), (86, 568)],
                "color": (205, 231, 235),
                "label_pos": (84, 604),
                "layer": 1,
            },
            {
                "quad": [(210, 584), (770, 584), (1092, 716), (42, 716)],
                "color": (224, 233, 245),
                "label_pos": (500, 756),
                "layer": 1,
            },
        ]

    if view_count >= 4 and view_angles and (projection_mode != "tsf" or view_count in (4, 8)):
        return _ring_like_specs()

    if projection_mode == "tsf":
        effective_primary_count = max(1, min(int(base_view_count or view_count or 1), 3))
        opposite_label_set = {"back", "left", "down"}
        is_opposite_triplet = bool(labels[:effective_primary_count]) and all(
            label in opposite_label_set for label in labels[:effective_primary_count]
        )
        specs = [
            {
                "quad": [(580, 120), (760, 170), (760, 500), (580, 455)] if is_opposite_triplet else [(95, 170), (260, 120), (260, 455), (95, 500)],
                "color": (199, 226, 245),
                "label_pos": (616, 534) if is_opposite_triplet else (94, 534),
                "layer": 1,
            },
            {
                "quad": [(95, 170), (260, 120), (260, 455), (95, 500)] if is_opposite_triplet else [(580, 120), (760, 170), (760, 500), (580, 455)],
                "color": (216, 198, 232),
                "label_pos": (94, 534) if is_opposite_triplet else (616, 534),
                "layer": 1,
            },
            {
                "quad": [(210, 500), (640, 500), (860, 582), (42, 582)] if is_opposite_triplet else [(210, 80), (640, 80), (860, 162), (42, 162)],
                "color": (224, 232, 245),
                "label_pos": (362, 618) if is_opposite_triplet else (362, 196),
                "layer": 0,
            },
        ]
        return specs[:effective_primary_count]

    return [
        {
            "quad": [(108, 185), (232, 118), (278, 430), (154, 498)],
            "color": (218, 198, 236),
            "label_pos": (122, 530),
            "layer": 1,
        },
        {
            "quad": [(420, 110), (565, 145), (565, 470), (420, 438)],
            "color": (228, 232, 182),
            "label_pos": (454, 502),
            "layer": 0,
        },
        {
            "quad": [(705, 145), (848, 110), (848, 438), (705, 470)],
            "color": (204, 220, 240),
            "label_pos": (724, 502),
            "layer": 1,
        },
    ]


def _create_plane_panel(
    hero_rgb: Optional[np.ndarray],
    silhouettes: Sequence[np.ndarray],
    silhouette_labels: Optional[Sequence[str]],
    projection_mode: str,
    title: str,
    subtitle: str,
    out_path: Path,
    include_extra_strip: bool = True,
    view_angles: Optional[Sequence[Tuple[float, float]]] = None,
    base_view_count: Optional[int] = None,
) -> str:
    plane_specs = _projection_plane_specs(
        projection_mode,
        len(silhouettes),
        silhouette_labels=silhouette_labels,
        view_angles=view_angles,
        base_view_count=base_view_count,
    )
    primary_count = len(plane_specs)
    extra_silhouettes = list(silhouettes[primary_count:]) if len(silhouettes) > primary_count else []
    extra_labels = list(silhouette_labels[primary_count:]) if silhouette_labels and len(silhouette_labels) > primary_count else []
    full_view_layout = len(plane_specs) >= 6
    width = 1180 if full_view_layout else 980
    height = 790 if full_view_layout else (760 if extra_silhouettes else 620)
    canvas = np.full((height, width, 3), (245, 246, 251), dtype=np.uint8)

    def draw_plane_spec(idx: int) -> None:
        quad = plane_specs[idx]["quad"]
        color = plane_specs[idx]["color"]
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, np.array(quad, dtype=np.int32), color)
        canvas[:] = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0.0)
        cv2.polylines(canvas, [np.array(quad, dtype=np.int32)], True, (178, 184, 196), 2, cv2.LINE_AA)
        if idx < len(silhouettes):
            silh = silhouettes[idx]
            if silh.ndim == 3:
                silh = cv2.cvtColor(silh, cv2.COLOR_BGR2GRAY)
            silh = cv2.resize(silh, (260, 260), interpolation=cv2.INTER_NEAREST)
            warped = _warp_mask_to_quad(silh, quad, (width, height))
            mask = warped > 127
            canvas[mask] = (70, 62, 78)
        label = silhouette_labels[idx] if silhouette_labels and idx < len(silhouette_labels) else None
        if label:
            label_x, label_y = plane_specs[idx]["label_pos"]
            cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (78, 80, 92), 2, cv2.LINE_AA)

    pre_hero_order = sorted(
        [idx for idx in range(len(plane_specs)) if int(plane_specs[idx].get("layer", 0)) < 2],
        key=lambda idx: int(plane_specs[idx].get("layer", 0)),
    )
    post_hero_order = sorted(
        [idx for idx in range(len(plane_specs)) if int(plane_specs[idx].get("layer", 0)) >= 2],
        key=lambda idx: int(plane_specs[idx].get("layer", 0)),
    )
    for idx in pre_hero_order:
        draw_plane_spec(idx)

    if hero_rgb is not None:
        hero_size = 390 if full_view_layout else 340
        hero = cv2.resize(hero_rgb, (hero_size, hero_size), interpolation=cv2.INTER_AREA)
        hero_x = (width - hero.shape[1]) // 2
        hero_y = 186 if full_view_layout else 178
        _overlay_foreground_bgr(canvas, hero, (hero_x, hero_y))

    for idx in post_hero_order:
        draw_plane_spec(idx)

    cv2.putText(canvas, title, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (34, 34, 44), 3, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (86, 88, 98), 2, cv2.LINE_AA)

    if include_extra_strip and extra_silhouettes:
        strip_title = "Mirrored / Opposite Silhouettes" if projection_mode == "tsf" else "Additional Paired Views"
        cv2.putText(canvas, strip_title, (32, 632), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (52, 54, 64), 2, cv2.LINE_AA)
        count = len(extra_silhouettes)
        gap = 18
        total_gap = gap * max(0, count - 1)
        card_w = min(178, max(136, (width - 64 - total_gap) // max(1, count)))
        card_h = 108
        strip_y = 648
        start_x = max(32, (width - (count * card_w + total_gap)) // 2)
        for idx, silh in enumerate(extra_silhouettes):
            x0 = start_x + idx * (card_w + gap)
            y0 = strip_y
            x1 = min(width - 20, x0 + card_w)
            y1 = min(height - 24, y0 + card_h)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 224, 232), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (196, 201, 211), 2, cv2.LINE_AA)
            tile = _resize_rgb(silh, (max(10, card_w - 18), max(10, card_h - 36)))
            tile_bgr = tile[..., ::-1]
            tile_h, tile_w = tile_bgr.shape[:2]
            tile_x0 = x0 + (card_w - tile_w) // 2
            tile_y0 = y0 + 8
            canvas[tile_y0:tile_y0 + tile_h, tile_x0:tile_x0 + tile_w] = tile_bgr
            label = extra_labels[idx] if idx < len(extra_labels) else f"view_{primary_count + idx}"
            cv2.putText(canvas, label, (x0 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (74, 76, 86), 2, cv2.LINE_AA)

    _ensure_parent(out_path)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def _split_projection_halves(
    silhouettes: Sequence[np.ndarray],
    silhouette_labels: Optional[Sequence[str]],
    base_view_count: int,
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
    primary_count = max(1, int(base_view_count))
    primary_silhouettes = list(silhouettes[:primary_count])
    primary_labels = list((silhouette_labels or [])[:primary_count])
    secondary_silhouettes = list(silhouettes[primary_count:])
    secondary_labels = list((silhouette_labels or [])[primary_count:])
    return primary_silhouettes, primary_labels, secondary_silhouettes, secondary_labels


def _compose_projection_panel_pair(
    primary_rgb: Optional[np.ndarray],
    secondary_rgb: Optional[np.ndarray],
    out_path: Path,
    gap: int = 26,
) -> str:
    primary = _resize_rgb(primary_rgb, (980, 620))
    if secondary_rgb is None:
        _ensure_parent(out_path)
        cv2.imwrite(str(out_path), cv2.cvtColor(primary, cv2.COLOR_RGB2BGR))
        return str(out_path)

    secondary = _resize_rgb(secondary_rgb, (980, 620))
    height = max(primary.shape[0], secondary.shape[0])
    width = primary.shape[1] + secondary.shape[1] + gap
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    canvas[:, :primary.shape[1]] = primary
    canvas[:, primary.shape[1] + gap:primary.shape[1] + gap + secondary.shape[1]] = secondary
    _ensure_parent(out_path)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return str(out_path)


def _create_projection_family_panels(
    hero_rgb: Optional[np.ndarray],
    silhouettes: Sequence[np.ndarray],
    silhouette_labels: Optional[Sequence[str]],
    projection_mode: str,
    primary_title: str,
    primary_subtitle: str,
    primary_out_path: Path,
    combined_out_path: Path,
    base_view_count: int,
    secondary_out_path: Optional[Path] = None,
    view_angles: Optional[Sequence[Tuple[float, float]]] = None,
    primary_hero_rgb: Optional[np.ndarray] = None,
    secondary_hero_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Optional[str]]:
    primary_silhs, primary_labels, secondary_silhs, secondary_labels = _split_projection_halves(
        silhouettes,
        silhouette_labels,
        base_view_count=base_view_count,
    )
    primary_angles = list(view_angles[:len(primary_silhs)]) if view_angles is not None else None
    secondary_angles = list(view_angles[len(primary_silhs):len(primary_silhs) + len(secondary_silhs)]) if view_angles is not None else None
    combined_subtitle = (
        "views: " + ", ".join(silhouette_labels)
        if silhouette_labels
        else primary_subtitle
    )
    _create_plane_panel(
        hero_rgb,
        list(silhouettes),
        list(silhouette_labels or []),
        projection_mode,
        primary_title,
        combined_subtitle,
        combined_out_path,
        include_extra_strip=False,
        view_angles=view_angles,
        base_view_count=base_view_count,
    )

    secondary_path_str = None
    primary_path_str = None
    if primary_silhs:
        _create_plane_panel(
            primary_hero_rgb if primary_hero_rgb is not None else hero_rgb,
            primary_silhs,
            primary_labels,
            projection_mode,
            primary_title,
            primary_subtitle,
            primary_out_path,
            include_extra_strip=False,
            view_angles=primary_angles,
            base_view_count=len(primary_silhs),
        )
        primary_path_str = str(primary_out_path)
    if secondary_silhs:
        secondary_title = (
            "Opposite Orthogonal Projection Sculpture"
            if projection_mode == "tsf"
            else "Paired Ring-View Sculpture"
        )
        secondary_subtitle = "views: " + ", ".join(secondary_labels) if secondary_labels else "paired / mirrored views"
        secondary_out_path = secondary_out_path or primary_out_path.with_name(primary_out_path.stem + "_opposite.png")
        _create_plane_panel(
            secondary_hero_rgb if secondary_hero_rgb is not None else hero_rgb,
            secondary_silhs,
            secondary_labels,
            projection_mode,
            secondary_title,
            secondary_subtitle,
            secondary_out_path,
            include_extra_strip=False,
            view_angles=secondary_angles,
            base_view_count=len(secondary_silhs),
        )
        secondary_path_str = str(secondary_out_path)
    return {
        "primary": primary_path_str,
        "secondary": secondary_path_str,
        "combined": str(combined_out_path),
    }


def _create_sixview_panel(summary: Dict[str, Any], hero_views: Sequence[np.ndarray], out_path: Path) -> str:
    expected_count = int(summary.get("view_count") or 0)
    labels = _get_view_labels(summary, view_limit=expected_count if expected_count > 0 else None)
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    n = min(len(labels), max(len(view_paths), len(hero_views), 0), expected_count if expected_count > 0 else 10 ** 9)
    if n == 0:
        labels = labels or ["view 0", "view 1", "view 2"]
        n = len(labels)

    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.2))
    axes = np.atleast_2d(axes)
    fig.suptitle("Model Views and Final Silhouettes", fontsize=18, fontweight="bold", y=0.98)
    subtitle = f"{summary['run_id']} | {summary['projection_mode']} | mean IoU={summary['aggregate_metrics']['mean_iou']:.4f}"
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11, color="#555555")

    for idx in range(n):
        render_img = hero_views[idx] if idx < len(hero_views) else None
        silh_img = _safe_read_gray(view_paths[idx]) if idx < len(view_paths) else None
        axes[0, idx].imshow(_resize_rgb(render_img, (280, 280)))
        axes[0, idx].set_title(labels[idx])
        axes[0, idx].axis("off")
        if silh_img is None:
            axes[1, idx].imshow(_resize_rgb(None, (280, 280)), cmap="gray")
        else:
            axes[1, idx].imshow(silh_img, cmap="gray")
        axes[1, idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _ensure_parent(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _compose_keyview_frame(render_rgb: Optional[np.ndarray], silh_gray: Optional[np.ndarray],
                           label: str, idx: int, total: int) -> np.ndarray:
    canvas = np.full((460, 860, 3), 247, dtype=np.uint8)
    cv2.rectangle(canvas, (28, 70), (548, 430), (220, 224, 232), 2)
    cv2.rectangle(canvas, (598, 120), (818, 340), (220, 224, 232), 2)

    render_panel = _resize_rgb(render_rgb, (500, 340))
    silh_panel = _resize_rgb(silh_gray, (200, 200))
    canvas[80:420, 38:538] = render_panel[..., ::-1]
    canvas[130:330, 608:808] = silh_panel[..., ::-1]

    cv2.putText(canvas, f"Silhouette Key Views  {idx + 1}/{total}", (28, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.88, (40, 40, 48), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"target: {label}", (28, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (84, 86, 96), 2, cv2.LINE_AA)
    cv2.putText(canvas, "model view", (38, 446),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (84, 86, 96), 2, cv2.LINE_AA)
    cv2.putText(canvas, "target silhouette", (608, 366),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (84, 86, 96), 2, cv2.LINE_AA)
    return canvas[..., ::-1]


def _blend_frames(prev_rgb: np.ndarray, next_rgb: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = prev_rgb.astype(np.float32) * (1.0 - alpha) + next_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _interpolate_angles(start: Tuple[float, float], end: Tuple[float, float], steps: int) -> List[Tuple[float, float]]:
    if steps <= 0:
        return []
    start_elev, start_azim = float(start[0]), float(start[1])
    end_elev, end_azim = float(end[0]), float(end[1])
    delta_azim = ((end_azim - start_azim + 180.0) % 360.0) - 180.0
    out = []
    for step in range(1, steps + 1):
        t = step / float(steps)
        elev = start_elev + (end_elev - start_elev) * t
        azim = start_azim + delta_azim * t
        out.append((elev, azim))
    return out


def _angle_distance(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    start_elev, start_azim = float(start[0]), float(start[1])
    end_elev, end_azim = float(end[0]), float(end[1])
    delta_azim = ((end_azim - start_azim + 180.0) % 360.0) - 180.0
    delta_elev = end_elev - start_elev
    return float(np.hypot(delta_elev, delta_azim))


def _create_silhouette_keyviews_gif(summary: Dict[str, Any], hero_views: Sequence[np.ndarray],
                                    out_path: Path, device: torch.device, fps: int = 10,
                                    transition_frames: int = 8, pause_seconds: float = 4.0,
                                    hold_frames: Optional[int] = None) -> Optional[str]:
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    expected_count = int(summary.get("view_count") or 0)
    n = min(
        len(hero_views),
        len(view_paths),
        expected_count if expected_count > 0 else 10 ** 9,
    )
    if n == 0:
        return None

    hold_frames = max(1, int(round(float(pause_seconds) * float(fps)))) if hold_frames is None else int(hold_frames)
    labels = _get_view_labels(summary, view_limit=n)
    angles = _get_view_angles(summary, view_limit=n)
    key_frames: List[np.ndarray] = []
    for idx in range(n):
        silh = _safe_read_gray(view_paths[idx])
        key_frames.append(_compose_keyview_frame(hero_views[idx], silh, labels[idx], idx, n))

    frames: List[np.ndarray] = []
    frames.extend([key_frames[0]] * max(1, hold_frames))
    for idx in range(1, n):
        angle_dist = _angle_distance(angles[idx - 1], angles[idx])
        segment_frames = max(1, int(round(angle_dist / 16.0)), int(transition_frames))
        interp_angles = _interpolate_angles(angles[idx - 1], angles[idx], segment_frames)
        transition_rgbs = _render_summary_views(
            summary,
            interp_angles,
            device=device,
            image_size=180,
            camera_mode=summary.get("camera_mode", "ortho"),
        )
        target_silh = _safe_read_gray(view_paths[idx])
        for rgb in transition_rgbs:
            frames.append(_compose_keyview_frame(rgb, target_silh, labels[idx], idx, n))
        frames.extend([key_frames[idx]] * max(1, hold_frames))

    _ensure_parent(out_path)
    imageio.mimsave(out_path, frames, duration=1.0 / max(1, fps), loop=0)
    return str(out_path)


def write_metrics_report(context: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, str]:
    output_dir = Path(summary["source_dir"])
    sample_id = int(summary["sample_id"])
    final_metrics = summary["aggregate_metrics"]
    renderer_metrics = context.get("renderer_metrics") or summary.get("renderer_metrics") or {}
    export_metrics = context.get("export_metrics") or _metric_block(summary["metrics_history"])
    config = summary.get("config", {})
    obj_path = summary["artifact_paths"].get("obj")
    verts_count, faces_count = _parse_obj_counts(obj_path)
    runtime = context.get("runtime_seconds")
    if runtime is None:
        runtime = summary["metrics_history"].get("final_metrics", {}).get("runtime", {}).get("train_seconds")
    post_export_seconds = context.get("post_export_seconds")
    phase_metrics = summary.get("phase_metrics", {})
    normal_consistency = final_metrics.get("normal_consistency")
    keyview_pause_seconds = context.get("keyview_pause_seconds", config.get("keyview_pause_seconds", "N/A"))
    history = summary["metrics_history"].get("history", [])
    last_raw_loss = None
    last_normalized_loss = None
    if history:
        for entry in reversed(history):
            if last_raw_loss is None and entry.get("raw_total_loss") is not None:
                last_raw_loss = float(entry["raw_total_loss"])
            if last_raw_loss is None and entry.get("total_loss") is not None:
                last_raw_loss = float(entry["total_loss"])
            if last_normalized_loss is None and entry.get("normalized_total_loss") is not None:
                last_normalized_loss = float(entry["normalized_total_loss"])
            if last_normalized_loss is None and entry.get("total_loss") is not None:
                last_normalized_loss = float(entry["total_loss"])
            if last_raw_loss is not None and last_normalized_loss is not None:
                break

    view_labels = _get_view_labels(summary)
    export_ious = export_metrics.get("view_ious", [])
    export_dices = export_metrics.get("view_dices", [])
    renderer_ious = renderer_metrics.get("view_ious", [])
    renderer_dices = renderer_metrics.get("view_dices", [])

    lines = [
        "=" * 72,
        "POST-TRAINING METRICS REPORT",
        "=" * 72,
        "",
        "[Run Configuration]",
        f"Run ID: {summary['run_id']}",
        f"Sample ID: {sample_id}",
        f"Geometry Type: {summary['geometry_type']}",
        f"Projection Mode: {summary['projection_mode']}",
        f"Camera Mode: {summary.get('camera_mode', 'ortho')}",
        f"Mirror Mode: {summary.get('mirror_mode', False)}",
        f"Export Strategy: {summary.get('export_strategy', 'unknown')}",
        f"Voxel Resolution: {config.get('voxel_res', 'N/A')}",
        f"Export Resolution: {config.get('export_res', config.get('export_resolution', 'N/A'))}",
        f"Image Size: {config.get('img_size', 'N/A')}",
        f"Num Input Views: {config.get('num_input_views', summary.get('base_view_count', 'N/A'))}",
        f"Num Total Views: {config.get('num_total_views', summary.get('view_count', 'N/A'))}",
        f"Keyview Pause Seconds: {keyview_pause_seconds}",
        "",
        "[Renderer Metrics]",
        f"Mean IOU:  {float(renderer_metrics.get('mean_iou', 0.0)):.6f}",
        f"Mean DICE: {float(renderer_metrics.get('mean_dice', 0.0)):.6f}",
    ]
    for idx in range(max(len(renderer_ious), len(renderer_dices))):
        label = view_labels[idx] if idx < len(view_labels) else f"view_{idx}"
        lines.append(
            f"  {label:>10s}: IOU={float(renderer_ious[idx]) if idx < len(renderer_ious) else float('nan'):.6f}  "
            f"DICE={float(renderer_dices[idx]) if idx < len(renderer_dices) else float('nan'):.6f}"
        )

    lines += [
        "",
        "[Final Export Metrics]",
        f"Mean IOU:  {float(export_metrics.get('mean_iou', 0.0)):.6f}",
        f"Mean DICE: {float(export_metrics.get('mean_dice', 0.0)):.6f}",
        f"Occupied Voxels: {export_metrics.get('occupied_voxels', 'N/A')}",
        f"Connected Components: {export_metrics.get('component_count', 'N/A')}",
        f"Largest Component Ratio: {float(export_metrics.get('largest_component_ratio', 0.0)):.6f}" if export_metrics.get('largest_component_ratio') is not None else "Largest Component Ratio: N/A",
        f"Bridge Voxels Added: {export_metrics.get('bridge_voxels_added', 'N/A')}",
    ]
    for idx in range(max(len(export_ious), len(export_dices))):
        label = view_labels[idx] if idx < len(view_labels) else f"view_{idx}"
        lines.append(
            f"  {label:>10s}: IOU={float(export_ious[idx]) if idx < len(export_ious) else float('nan'):.6f}  "
            f"DICE={float(export_dices[idx]) if idx < len(export_dices) else float('nan'):.6f}"
        )

    lines += [
        "",
        "[Phase Best IOUs]",
    ]
    if phase_metrics:
        for key, value in phase_metrics.items():
            lines.append(f"  {key}: {float(value):.6f}")
    else:
        lines.append("  N/A")

    lines += [
        "",
        "[Geometry / Runtime]",
        f"OBJ Vertices: {verts_count if verts_count is not None else 'N/A'}",
        f"OBJ Faces: {faces_count if faces_count is not None else 'N/A'}",
        f"Last Raw Total Loss: {'N/A' if last_raw_loss is None else f'{last_raw_loss:.6f}'}",
        f"Last Normalized Loss: {'N/A' if last_normalized_loss is None else f'{last_normalized_loss:.6f}'}",
        f"Normal Consistency Loss: {'N/A' if normal_consistency is None else f'{float(normal_consistency):.6f}'}",
        f"Training Time: {_seconds_to_hms(runtime)}",
        f"Post Export Time: {_seconds_to_hms(post_export_seconds)}",
        "=" * 72,
        "",
    ]

    report_text = "\n".join(lines)
    sample_report_path = output_dir / f"sample_{sample_id}_metrics_report.txt"
    default_report_path = output_dir / "metrics_report.txt"
    _save_text(sample_report_path, report_text)
    _save_text(default_report_path, report_text)
    return {"sample_report": str(sample_report_path), "default_report": str(default_report_path)}


def _hero_render(summary: Dict[str, Any], device: torch.device, azim: float = 35.0, elev: float = 20.0,
                 camera_mode: Optional[str] = None, image_size: int = 280) -> Optional[np.ndarray]:
    imgs = _render_summary_views(
        summary,
        [(elev, azim)],
        device=device,
        image_size=image_size,
        camera_mode=camera_mode or summary.get("camera_mode", "ortho"),
    )
    return imgs[0] if imgs else None


def generate_current_sample_visuals(context: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Optional[str]]:
    device = _resolve_device(context.get("device"))
    output_dir = Path(summary["source_dir"])
    sample_id = int(summary["sample_id"])
    hero = _hero_render(summary, device)
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    expected_view_count = int(summary.get("view_count") or len(view_paths) or 0)
    render_view_limit = min(
        len(view_paths),
        expected_view_count if expected_view_count > 0 else len(view_paths),
    )
    view_angles = _get_view_angles(summary, view_limit=render_view_limit)
    view_renders = _render_summary_views_aligned_to_targets(summary, view_paths[:render_view_limit], device, image_size=240)
    if not view_renders:
        view_renders = _render_summary_views(summary, view_angles[:render_view_limit], device, image_size=240, camera_mode=summary.get("camera_mode", "ortho"))
    artistic_view_limit = render_view_limit
    silhouette_labels = _get_view_labels(summary, view_limit=artistic_view_limit)
    if summary.get("projection_mode") == "tsf":
        silhouettes = _ordered_illustration_masks(summary, silhouette_labels)
    else:
        silhouettes = []
        for idx in range(artistic_view_limit):
            silh = _safe_read_gray(view_paths[idx])
            if silh is not None:
                silhouettes.append(silh)
    artistic_view_angles = _get_view_angles(summary, view_limit=artistic_view_limit)
    base_view_count = int(summary.get("base_view_count") or (artistic_view_limit // 2 if summary.get("mirror_mode") and artistic_view_limit % 2 == 0 else min(3, artistic_view_limit)))

    artistic_primary_path = output_dir / f"sample_{sample_id}_projection_artistic_primary.png"
    artistic_opposite_path = output_dir / f"sample_{sample_id}_projection_artistic_opposite.png"
    artistic_path = output_dir / f"sample_{sample_id}_projection_artistic.png"
    artistic_title = "Orthogonal Projection Sculpture" if summary["projection_mode"] == "tsf" else "Ring-View Projection Sculpture"
    primary_labels = silhouette_labels[:base_view_count]
    artistic_subtitle = (
        "views: " + ", ".join(primary_labels)
        if primary_labels
        else f"{summary['projection_mode']} | export={summary.get('export_strategy', 'unknown')}"
    )
    artistic_paths = _create_projection_family_panels(
        hero_rgb=hero,
        silhouettes=silhouettes,
        silhouette_labels=silhouette_labels,
        projection_mode=summary["projection_mode"],
        primary_title=artistic_title,
        primary_subtitle=artistic_subtitle,
        primary_out_path=artistic_primary_path,
        secondary_out_path=artistic_opposite_path,
        combined_out_path=artistic_path,
        base_view_count=base_view_count,
        view_angles=artistic_view_angles,
        primary_hero_rgb=view_renders[0] if view_renders else hero,
        secondary_hero_rgb=view_renders[base_view_count] if len(view_renders) > base_view_count else hero,
    )

    sixview_path = output_dir / f"sample_{sample_id}_projection_sixview.png"
    _create_sixview_panel(summary, view_renders, sixview_path)
    keyviews_gif_path = output_dir / f"sample_{sample_id}_silhouette_keyviews.gif"
    keyviews_gif = _create_silhouette_keyviews_gif(
        summary,
        view_renders,
        keyviews_gif_path,
        device=device,
        fps=max(6, int(context.get("rotation_fps", 18))),
        transition_frames=8,
        pause_seconds=float(context.get("keyview_pause_seconds", 10.0)),
    )

    return {
        "artistic_panel": str(artistic_path),
        "artistic_panel_primary": artistic_paths["primary"],
        "artistic_panel_opposite": artistic_paths["secondary"],
        "sixview_panel": str(sixview_path),
        "keyviews_gif": keyviews_gif,
    }


def _tile_view_strip(summary: Dict[str, Any], limit: int = 3, tile_size: Tuple[int, int] = (180, 180)) -> np.ndarray:
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    limit = min(limit, max(len(view_paths), 1))
    labels = _get_view_labels(summary, view_limit=limit)
    tiles = []
    for idx in range(limit):
        img = _safe_read_gray(view_paths[idx]) if idx < len(view_paths) else None
        tile = _resize_rgb(img, tile_size)
        tile_bgr = tile[..., ::-1].copy()
        cv2.putText(tile_bgr, labels[idx], (12, tile_bgr.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (48, 48, 48), 2, cv2.LINE_AA)
        tiles.append(tile_bgr[..., ::-1])
    return np.concatenate(tiles, axis=1)


def _tile_view_grid(summary: Dict[str, Any], limit: int = 6, tile_size: Tuple[int, int] = (180, 180),
                    ncols: int = 3, gap: int = 14) -> np.ndarray:
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    limit = min(limit, max(len(view_paths), 1))
    labels = _get_view_labels(summary, view_limit=limit)
    tile_w, tile_h = tile_size
    ncols = max(1, min(ncols, limit))
    nrows = int(math.ceil(limit / ncols))
    canvas_h = nrows * tile_h + max(0, nrows - 1) * gap
    canvas_w = ncols * tile_w + max(0, ncols - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)

    for idx in range(limit):
        img = _safe_read_gray(view_paths[idx]) if idx < len(view_paths) else None
        tile = _resize_rgb(img, tile_size)
        tile_bgr = tile[..., ::-1].copy()
        cv2.rectangle(tile_bgr, (0, tile_h - 34), (tile_w, tile_h), (236, 238, 243), -1)
        cv2.putText(tile_bgr, labels[idx], (10, tile_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (48, 48, 48), 2, cv2.LINE_AA)
        row, col = divmod(idx, ncols)
        y0 = row * (tile_h + gap)
        x0 = col * (tile_w + gap)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile_bgr
    return canvas[..., ::-1]


def _placeholder_image(title: str, subtitle: str, size: Tuple[int, int] = (900, 520)) -> np.ndarray:
    w, h = size
    canvas = np.full((h, w, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (32, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (44, 44, 52), 3, cv2.LINE_AA)
    wrapped = textwrap.wrap(subtitle, width=48)
    y = 130
    for line in wrapped:
        cv2.putText(canvas, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (108, 108, 118), 2, cv2.LINE_AA)
        y += 34
    cv2.rectangle(canvas, (48, 180), (w - 48, h - 48), (210, 214, 222), 2)
    cv2.putText(canvas, "PLACEHOLDER", (w // 2 - 122, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (160, 162, 170), 3, cv2.LINE_AA)
    return canvas[..., ::-1]


def _save_paper_figure(fig_path: Path, image: np.ndarray) -> str:
    _ensure_parent(fig_path)
    if image.ndim == 3 and image.shape[2] == 3:
        cv2.imwrite(str(fig_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(fig_path), image)
    return str(fig_path)


def _draw_plane_mask(
    canvas_bgr: np.ndarray,
    quad: Sequence[Tuple[int, int]],
    mask_gray: Optional[np.ndarray],
    fill_color: Tuple[int, int, int],
    mask_color: Tuple[int, int, int] = (28, 28, 36),
    alpha: float = 0.22,
) -> None:
    overlay = canvas_bgr.copy()
    cv2.fillConvexPoly(overlay, np.array(quad, dtype=np.int32), fill_color)
    canvas_bgr[:] = cv2.addWeighted(overlay, alpha, canvas_bgr, 1.0 - alpha, 0.0)
    cv2.polylines(canvas_bgr, [np.array(quad, dtype=np.int32)], True, (194, 198, 206), 2, cv2.LINE_AA)
    if mask_gray is not None:
        warped = _warp_mask_to_quad(mask_gray, quad, (canvas_bgr.shape[1], canvas_bgr.shape[0]))
        canvas_bgr[warped > 127] = mask_color


def _draw_callout(
    canvas_bgr: np.ndarray,
    text: str,
    text_pos: Tuple[int, int],
    anchor: Tuple[int, int],
    font_scale: float = 0.72,
) -> None:
    cv2.putText(canvas_bgr, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (46, 46, 54), 2, cv2.LINE_AA)
    line_start = (text_pos[0] + 8, text_pos[1] - 10)
    cv2.arrowedLine(canvas_bgr, line_start, anchor, (76, 78, 88), 2, cv2.LINE_AA, tipLength=0.035)


def _load_reference_masks(summary: Dict[str, Any], limit: int = 6) -> List[np.ndarray]:
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    masks: List[np.ndarray] = []
    for path in view_paths[:limit]:
        mask = _safe_read_gray(path)
        if mask is not None:
            masks.append(mask)
    return masks


def _load_named_masks(summary: Dict[str, Any], limit: int = 6) -> Dict[str, np.ndarray]:
    view_paths = summary["artifact_paths"].get("final_views") or summary["artifact_paths"].get("pred_cleaned_views") or summary["artifact_paths"].get("renderer_views") or []
    labels = _get_view_labels(summary, view_limit=min(limit, len(view_paths)))
    named: Dict[str, np.ndarray] = {}
    for idx, path in enumerate(view_paths[:limit]):
        mask = _safe_read_gray(path)
        if mask is None or idx >= len(labels):
            continue
        named[str(labels[idx])] = mask
    return named


def _flip_mask_lr(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return np.ascontiguousarray(np.fliplr(mask))


def _load_illustration_named_masks(summary: Dict[str, Any], limit: int = 6) -> Dict[str, np.ndarray]:
    named = _load_named_masks(summary, limit=limit)
    if summary.get("projection_mode") != "tsf":
        return named

    # 对论文示意图来说，镜像/对向视图优先使用主视图的水平翻转，
    # 这样 front/back、right/left、top/down 的“成对关系”会更直观。
    for opposite, primary in (("back", "front"), ("left", "right"), ("down", "top")):
        primary_mask = named.get(primary)
        flipped_primary = _flip_mask_lr(primary_mask)
        if flipped_primary is not None:
            named[opposite] = flipped_primary
    return named


def _ordered_illustration_masks(summary: Dict[str, Any], labels: Sequence[str]) -> List[np.ndarray]:
    named = _load_illustration_named_masks(summary, limit=max(6, len(labels)))
    ordered: List[np.ndarray] = []
    for label in labels:
        mask = named.get(label)
        if mask is not None:
            ordered.append(mask)
    return ordered


def _build_teaser_figure(current_summary: Dict[str, Any], all_summaries: Sequence[Dict[str, Any]], fig_path: Path,
                         manifest_entry: Dict[str, Any], device: torch.device) -> None:
    best_mesh = _pick_best_summary(all_summaries, geometry_type="mesh")
    best_voxel = current_summary if current_summary.get("geometry_type") == "voxel" else (_pick_best_summary(all_summaries, geometry_type="voxel") or current_summary)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Paper Figure 01 - Teaser", fontsize=20, fontweight="bold", y=0.98)
    chosen = [best_mesh, best_voxel]
    column_titles = ["Mesh 3D Sculpture", "Voxel 3D Sculpture"]
    sources = []
    status = "exact"

    for col, summary in enumerate(chosen):
        if summary is None:
            status = "placeholder"
            placeholder = _placeholder_image(column_titles[col], "No historical sample available.")
            axes[0, col].imshow(placeholder)
            axes[1, col].imshow(placeholder)
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        hero = _hero_render(summary, device)
        view_limit = 6 if len(summary["artifact_paths"].get("final_views") or []) >= 6 else 3
        strip = _tile_view_grid(summary, limit=view_limit, tile_size=(165, 165), ncols=3) if view_limit > 3 else _tile_view_strip(summary, limit=3)
        axes[0, col].imshow(_resize_rgb(hero, (420, 420)))
        axes[0, col].set_title(f"{column_titles[col]}\n{summary['run_id']}", fontsize=14)
        axes[1, col].imshow(strip)
        axes[1, col].set_title(
            f"Mean IoU={summary['aggregate_metrics']['mean_iou']:.4f} | "
            f"Mean Dice={summary['aggregate_metrics']['mean_dice']:.4f}",
            fontsize=12,
        )
        axes[0, col].axis("off")
        axes[1, col].axis("off")
        sources.append(summary["source_dir"])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": status if best_mesh and best_voxel else "adapted",
        "sources": sources,
        "notes": "Representative mesh and voxel samples drawn from the best available runs.",
    })


def _build_projection_setup_figure(current_summary: Dict[str, Any], all_summaries: Sequence[Dict[str, Any]], fig_path: Path,
                                   manifest_entry: Dict[str, Any], device: torch.device) -> None:
    tsf_summary = current_summary if current_summary.get("projection_mode") == "tsf" else (_pick_best_summary(all_summaries, projection_mode="tsf") or current_summary)
    ring_summary = current_summary if current_summary.get("projection_mode") == "ring" else _pick_best_summary(all_summaries, projection_mode="ring")
    status = "exact" if ring_summary is not None else "adapted"

    left = None
    right = None

    if tsf_summary:
        hero = _hero_render(tsf_summary, device)
        views = tsf_summary["artifact_paths"].get("final_views") or []
        view_limit = min(len(views), 6)
        silhouette_labels = _get_view_labels(tsf_summary, view_limit=view_limit)
        silhouettes = _ordered_illustration_masks(tsf_summary, silhouette_labels)
        silhouette_angles = _get_view_angles(tsf_summary, view_limit=view_limit)
        base_view_count = int(tsf_summary.get("base_view_count") or 3)
        tmp_primary = fig_path.parent / "_tmp_tsf_setup_primary.png"
        tmp_secondary = fig_path.parent / "_tmp_tsf_setup_opposite.png"
        tmp_combined = fig_path.parent / "_tmp_tsf_setup.png"
        _create_projection_family_panels(
            hero_rgb=hero,
            silhouettes=silhouettes,
            silhouette_labels=silhouette_labels,
            projection_mode="tsf",
            primary_title="TSF Orthogonal Setup",
            primary_subtitle="views: " + ", ".join(silhouette_labels[:base_view_count]),
            primary_out_path=tmp_primary,
            secondary_out_path=tmp_secondary,
            combined_out_path=tmp_combined,
            base_view_count=base_view_count,
            view_angles=silhouette_angles,
        )
        left = _safe_read_rgb(str(tmp_combined))
        for tmp in (tmp_primary, tmp_secondary, tmp_combined):
            if tmp.exists():
                tmp.unlink()

    if ring_summary:
        hero = _hero_render(ring_summary, device)
        views = ring_summary["artifact_paths"].get("final_views") or []
        view_limit = min(len(views), 6)
        silhouettes = [img for img in (_safe_read_gray(p) for p in views[:view_limit]) if img is not None]
        silhouette_labels = _get_view_labels(ring_summary, view_limit=view_limit)
        silhouette_angles = _get_view_angles(ring_summary, view_limit=view_limit)
        base_view_count = int(ring_summary.get("base_view_count") or max(1, view_limit // 2))
        tmp_primary = fig_path.parent / "_tmp_ring_setup_primary.png"
        tmp_secondary = fig_path.parent / "_tmp_ring_setup_opposite.png"
        tmp_combined = fig_path.parent / "_tmp_ring_setup.png"
        _create_projection_family_panels(
            hero_rgb=hero,
            silhouettes=silhouettes,
            silhouette_labels=silhouette_labels,
            projection_mode="ring",
            primary_title="Non-TSF Ring Setup",
            primary_subtitle="views: " + ", ".join(silhouette_labels[:base_view_count]),
            primary_out_path=tmp_primary,
            secondary_out_path=tmp_secondary,
            combined_out_path=tmp_combined,
            base_view_count=base_view_count,
            view_angles=silhouette_angles,
        )
        right = _safe_read_rgb(str(tmp_combined))
        for tmp in (tmp_primary, tmp_secondary, tmp_combined):
            if tmp.exists():
                tmp.unlink()
    else:
        right = _placeholder_image("Non-TSF Setup", "No historical non-TSF run found. Layout shown as adapted placeholder.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Paper Figure 02 - Projection Setups", fontsize=18, fontweight="bold", y=0.98)
    axes[0].imshow(_resize_rgb(left, (760, 480)))
    axes[0].set_title("TSF / Orthogonal Projection")
    axes[1].imshow(_resize_rgb(right, (760, 480)))
    axes[1].set_title("Non-TSF / Ring-View Projection")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": status,
        "sources": [s for s in [tsf_summary["source_dir"] if tsf_summary else None, ring_summary["source_dir"] if ring_summary else None] if s],
        "notes": "Projection setup panels follow the current camera definitions. Missing non-TSF history falls back to an adapted panel.",
    })


def _build_pipeline_figure(current_summary: Dict[str, Any], fig_path: Path,
                           manifest_entry: Dict[str, Any], device: torch.device) -> None:
    canvas = np.full((760, 1180, 3), 250, dtype=np.uint8)
    named_masks = _load_illustration_named_masks(current_summary, limit=6)
    if not named_masks:
        placeholder = _placeholder_image("Pipeline", "No silhouettes available for pipeline illustration.")
        _save_paper_figure(fig_path, placeholder)
        manifest_entry.update({
            "status": "placeholder",
            "sources": [current_summary["source_dir"]],
            "notes": "No silhouettes were available to draw the pipeline schematic.",
        })
        return

    hero = _hero_render(current_summary, device)
    light_points = [(120, 120), (590, 72), (1060, 120)]
    cone_targets = [(240, 505), (590, 565), (940, 505)]
    cone_colors = [(250, 222, 222), (225, 244, 221), (223, 228, 250)]
    for light_pt, target_pt, cone_color in zip(light_points, cone_targets, cone_colors):
        overlay = canvas.copy()
        cv2.fillConvexPoly(
            overlay,
            np.array([light_pt, (target_pt[0] - 88, target_pt[1] - 72), (target_pt[0] + 88, target_pt[1] - 72)], dtype=np.int32),
            cone_color,
        )
        canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0)
        cv2.circle(canvas, light_pt, 6, (92, 94, 104), -1, cv2.LINE_AA)

    semantic_quads = {
        "front": [(194, 188), (258, 152), (258, 330), (194, 366)],
        "top": [(544, 154), (636, 154), (636, 314), (544, 314)],
        "right": [(898, 152), (962, 188), (962, 366), (898, 330)],
        "left": [(162, 470), (252, 420), (252, 634), (162, 684)],
        "down": [(500, 496), (678, 496), (678, 684), (500, 684)],
        "back": [(930, 420), (1020, 470), (1020, 684), (930, 634)],
    }
    semantic_colors = {
        "front": (236, 238, 244),
        "top": (236, 242, 248),
        "right": (244, 238, 246),
        "left": (244, 244, 244),
        "down": (244, 244, 244),
        "back": (244, 244, 244),
    }
    for key in ["front", "top", "right", "left", "down", "back"]:
        _draw_plane_mask(canvas, semantic_quads[key], named_masks.get(key), semantic_colors[key])

    if hero is not None:
        hero_small = _resize_rgb(hero, (230, 230))
        _overlay_foreground_bgr(canvas, hero_small, (474, 282))
        cv2.rectangle(canvas, (510, 310), (670, 520), (196, 202, 210), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Paper Figure 03 - Shadow Hull Pipeline", (34, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.12, (32, 32, 40), 3, cv2.LINE_AA)
    _draw_callout(canvas, "input images", (362, 168), (592, 212))
    _draw_callout(canvas, "light sources", (760, 126), (1060, 120))
    _draw_callout(canvas, "shadow cone", (964, 242), (928, 418))
    _draw_callout(canvas, "shadow hull", (76, 392), (514, 438))
    _draw_callout(canvas, "actual shadow", (110, 716), (588, 646))
    _draw_callout(canvas, "desired shadow", (884, 716), (972, 556))

    image = canvas[..., ::-1]
    _save_paper_figure(fig_path, image)
    manifest_entry.update({
        "status": "adapted",
        "sources": [current_summary["source_dir"]],
        "notes": "Adapted pipeline schematic styled after the reference shadow-hull illustration, using current run silhouettes.",
    })


def _build_ablation_figure(current_summary: Dict[str, Any], current_context: Dict[str, Any], fig_path: Path,
                           manifest_entry: Dict[str, Any], device: torch.device) -> None:
    canvas = np.full((760, 980, 3), 250, dtype=np.uint8)
    named_masks = _load_illustration_named_masks(current_summary, limit=6)
    if not named_masks:
        placeholder = _placeholder_image("Least-Cost Voxel", "No silhouettes were available for the voxel-cost illustration.")
        _save_paper_figure(fig_path, placeholder)
        manifest_entry.update({
            "status": "placeholder",
            "sources": [current_summary["source_dir"]],
            "notes": "No silhouettes were available to draw the least-cost voxel illustration.",
        })
        return

    hero = _hero_render(current_summary, device)
    semantic_specs = {
        "front": ([(166, 196), (242, 156), (242, 326), (166, 366)], (237, 236, 244), (156, 396)),
        "right": ([(748, 156), (824, 196), (824, 366), (748, 326)], (236, 244, 247), (742, 396)),
        "top": ([(338, 78), (606, 78), (678, 126), (266, 126)], (232, 246, 228), (286, 148)),
        "left": ([(154, 488), (230, 448), (230, 620), (154, 660)], (241, 238, 246), (144, 694)),
        "down": ([(338, 642), (606, 642), (678, 690), (266, 690)], (238, 239, 247), (278, 724)),
        "back": ([(748, 448), (824, 488), (824, 660), (748, 620)], (243, 236, 231), (742, 694)),
    }

    for key in ["front", "right", "top", "left", "down", "back"]:
        quad, color, label_pos = semantic_specs[key]
        _draw_plane_mask(canvas, quad, named_masks.get(key), color)
        cv2.putText(canvas, key, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (52, 52, 60), 2, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (446, 150), (534, 610), (219, 239, 222), -1)
    cv2.rectangle(overlay, (330, 298), (650, 384), (239, 222, 234), -1)
    cv2.rectangle(overlay, (392, 212), (588, 540), (224, 232, 246), -1)
    canvas = cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0.0)
    cv2.rectangle(canvas, (446, 150), (534, 610), (170, 198, 174), 2)
    cv2.rectangle(canvas, (330, 298), (650, 384), (198, 170, 184), 2)
    cv2.rectangle(canvas, (392, 212), (588, 540), (170, 182, 198), 2)

    if hero is not None:
        hero_small = _resize_rgb(hero, (170, 170))
        _overlay_foreground_bgr(canvas, hero_small, (406, 260))

    least_cost_pt = (490, 214)
    cv2.circle(canvas, least_cost_pt, 8, (46, 46, 52), -1, cv2.LINE_AA)
    cv2.circle(canvas, (650, 298), 6, (70, 70, 78), -1, cv2.LINE_AA)

    bar_x0, bar_y0 = 84, 650
    for idx in range(220):
        value = int(round(255.0 * idx / 219.0))
        cv2.line(canvas, (bar_x0 + idx, bar_y0), (bar_x0 + idx, bar_y0 + 24), (value, value, value), 1)
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x0 + 220, bar_y0 + 24), (40, 40, 48), 1)

    cv2.putText(canvas, "Paper Figure 04 - Least-Cost Voxel Illustration", (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (32, 32, 40), 3, cv2.LINE_AA)
    _draw_callout(canvas, "least-cost voxel", (542, 126), least_cost_pt)
    _draw_callout(canvas, "positional constraint", (716, 258), (650, 298))
    cv2.putText(canvas, "voxel cost", (84, 638), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (46, 46, 54), 2, cv2.LINE_AA)
    cv2.putText(canvas, "min", (84, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (72, 72, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, "max", (270, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (72, 72, 80), 2, cv2.LINE_AA)

    image = canvas[..., ::-1]
    _save_paper_figure(fig_path, image)
    manifest_entry.update({
        "status": "adapted",
        "sources": [current_summary["source_dir"]],
        "notes": "Adapted schematic styled after the least-cost voxel illustration, using current run silhouettes.",
    })


def _comparison_card(ax, summary: Optional[Dict[str, Any]], title: str, device: torch.device, note_if_missing: str) -> Tuple[str, Optional[str]]:
    if summary is None:
        ax.imshow(_placeholder_image(title, note_if_missing))
        ax.axis("off")
        return "placeholder", None

    hero = _hero_render(summary, device)
    strip = _tile_view_strip(summary, limit=min(summary.get("base_view_count", 3), 3))
    card = np.full((760, 540, 3), 250, dtype=np.uint8)
    hero_small = _resize_rgb(hero, (480, 360))
    strip_small = cv2.resize(strip[..., ::-1], (480, 150), interpolation=cv2.INTER_AREA)[..., ::-1]
    card[28:388, 30:510] = hero_small
    card[418:568, 30:510] = strip_small
    card = card[..., ::-1].copy()
    cv2.putText(card, title, (28, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (45, 45, 52), 2, cv2.LINE_AA)
    cv2.putText(card, f"run={summary['run_id']}", (28, 604), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (72, 72, 80), 2, cv2.LINE_AA)
    cv2.putText(card, f"mean IoU={summary['aggregate_metrics']['mean_iou']:.4f}", (28, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (72, 72, 80), 2, cv2.LINE_AA)
    cv2.putText(card, f"mean Dice={summary['aggregate_metrics']['mean_dice']:.4f}", (28, 676), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (72, 72, 80), 2, cv2.LINE_AA)
    ax.imshow(card[..., ::-1])
    ax.axis("off")
    return "exact", summary["source_dir"]


def _build_projection_comparison_figure(current_summary: Dict[str, Any], all_summaries: Sequence[Dict[str, Any]], fig_path: Path,
                                        manifest_entry: Dict[str, Any], device: torch.device) -> None:
    two_view = _pick_best_summary(all_summaries, base_view_count=2)
    tsf = _pick_best_summary(all_summaries, projection_mode="tsf") or (current_summary if current_summary.get("projection_mode") == "tsf" else None)
    ring = _pick_best_summary(all_summaries, projection_mode="ring")

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.2))
    fig.suptitle("Paper Figure 05 - Projection Regimes", fontsize=18, fontweight="bold", y=0.98)
    statuses = []
    sources = []
    for ax, summary, title, note in (
        (axes[0], two_view, "Two Views", "No 2-view run found; card left as placeholder."),
        (axes[1], tsf, "Three Orthogonal Views", "No TSF run found; card left as placeholder."),
        (axes[2], ring, "Three Non-Orthogonal Views", "No non-TSF run found; card left as placeholder."),
    ):
        status, src = _comparison_card(ax, summary, title, device, note)
        statuses.append(status)
        if src:
            sources.append(src)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": "exact" if all(s == "exact" for s in statuses) else "adapted",
        "sources": sources,
        "notes": "Automatically aggregates the best available runs for each projection regime.",
    })


def _build_prior_comparison_figure(current_summary: Dict[str, Any], all_summaries: Sequence[Dict[str, Any]], fig_path: Path,
                                   manifest_entry: Dict[str, Any], device: torch.device) -> None:
    prior = _pick_best_summary([s for s in all_summaries if s["source_project"] == "mesh_combined"], geometry_type="mesh")
    current = current_summary
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Paper Figure 06 - Prior Mesh vs Current Voxel", fontsize=18, fontweight="bold", y=0.98)

    statuses = []
    sources = []
    for ax, summary, title in (
        (axes[0], prior, "Historical Mesh Pipeline"),
        (axes[1], current, "Current Voxel Pipeline"),
    ):
        if summary is None:
            ax.imshow(_placeholder_image(title, "Historical comparison sample is unavailable."))
            statuses.append("placeholder")
        else:
            hero = _hero_render(summary, device)
            strip = _tile_view_strip(summary, limit=3)
            card = np.full((760, 560, 3), 250, dtype=np.uint8)
            card[30:390, 40:520] = _resize_rgb(hero, (480, 360))
            strip_small = cv2.resize(strip[..., ::-1], (480, 155), interpolation=cv2.INTER_AREA)[..., ::-1]
            card[420:575, 40:520] = strip_small
            ax.imshow(card)
            ax.set_title(f"{title}\n{summary['run_id']}", fontsize=13)
            statuses.append("exact")
            sources.append(summary["source_dir"])
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": "adapted" if prior is not None else "placeholder",
        "sources": sources,
        "notes": "Uses the best available historical mesh run as a stand-in for prior-method comparison.",
    })


def _build_voxel_gallery_figure(current_summary: Dict[str, Any], fig_path: Path, manifest_entry: Dict[str, Any], device: torch.device) -> None:
    hero_a = _hero_render(current_summary, device, azim=35.0, elev=20.0)
    hero_b = _hero_render(current_summary, device, azim=145.0, elev=20.0)
    strip = _tile_view_strip(current_summary, limit=3)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Paper Figure 07 - Voxel Sculpture Gallery", fontsize=18, fontweight="bold", y=0.98)
    axes[0, 0].imshow(_resize_rgb(hero_a, (520, 420)))
    axes[0, 0].set_title("Hero View")
    axes[0, 1].imshow(_resize_rgb(hero_b, (520, 420)))
    axes[0, 1].set_title("Alternate View")
    axes[1, 0].imshow(strip)
    axes[1, 0].set_title("Silhouette Strip")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5, 0.6,
        f"Export strategy: {current_summary.get('export_strategy', 'unknown')}\n"
        f"Occupied voxels: {current_summary['aggregate_metrics'].get('occupied_voxels', 'N/A')}\n"
        f"Mean IoU: {current_summary['aggregate_metrics']['mean_iou']:.4f}\n"
        f"Mean Dice: {current_summary['aggregate_metrics']['mean_dice']:.4f}",
        ha="center", va="center", fontsize=13,
        bbox=dict(boxstyle="round", facecolor="#ECF2FA", edgecolor="#98A2B3"),
    )
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": "exact",
        "sources": [current_summary["source_dir"]],
        "notes": "Focused gallery for the current voxel export.",
    })


def _build_sketch_gallery_figure(current_summary: Dict[str, Any], fig_path: Path, manifest_entry: Dict[str, Any], device: torch.device) -> None:
    gt_paths = current_summary["artifact_paths"].get("gt_views") or current_summary["artifact_paths"].get("final_views") or []
    inputs = []
    for path in gt_paths[:3]:
        img = _safe_read_gray(path)
        if img is None:
            continue
        edge = cv2.Canny(img, 50, 150)
        inputs.append(edge)

    hero = _hero_render(current_summary, device, azim=35.0, elev=20.0)
    alt = _hero_render(current_summary, device, azim=215.0, elev=18.0)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    fig.suptitle("Paper Figure 08 - Sketch-Like Input Adaptation", fontsize=18, fontweight="bold", y=0.98)

    for idx in range(3):
        if idx < len(inputs):
            axes[0, idx].imshow(inputs[idx], cmap="gray")
            axes[0, idx].set_title(f"Input sketch-like view {idx}")
        else:
            axes[0, idx].imshow(_placeholder_image("Missing Input", "No silhouette found for this slot."))
        axes[0, idx].axis("off")

    axes[1, 0].imshow(_resize_rgb(hero, (350, 300)))
    axes[1, 0].set_title("Reconstruction Hero View")
    axes[1, 1].imshow(_resize_rgb(alt, (350, 300)))
    axes[1, 1].set_title("Alternate View")
    axes[1, 2].imshow(_tile_view_strip(current_summary, limit=3))
    axes[1, 2].set_title("Output Silhouettes")
    for ax in axes[1]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": "adapted",
        "sources": [current_summary["source_dir"]],
        "notes": "Adapts silhouette inputs into sketch-like edge views because original sketch datasets are not present locally.",
    })


def _build_printable_gallery_figure(current_summary: Dict[str, Any], fig_path: Path, manifest_entry: Dict[str, Any], device: torch.device) -> None:
    views = _render_summary_views(
        current_summary,
        [(18.0, 0.0), (18.0, 90.0), (18.0, 180.0), (18.0, 270.0)],
        device,
        image_size=220,
        camera_mode=current_summary.get("camera_mode", "ortho"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 10))
    fig.suptitle("Paper Figure 09 - Printable Gallery", fontsize=18, fontweight="bold", y=0.98)
    for idx, ax in enumerate(axes.flat):
        img = views[idx] if idx < len(views) else None
        ax.imshow(_resize_rgb(img, (360, 360)))
        ax.set_title(f"Rotation {idx * 90}°")
        ax.axis("off")

    fig.text(
        0.5, 0.02,
        "Rendered substitute for printable-gallery figure. Local environment has no physical print photos.",
        ha="center", va="center", fontsize=11, color="#666666",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(fig_path)
    plt.close(fig)
    manifest_entry.update({
        "status": "adapted",
        "sources": [current_summary["source_dir"]],
        "notes": "Uses rendered substitute views when real 3D-print photos are unavailable.",
    })


def generate_paper_figures(current_summary: Dict[str, Any], all_summaries: Sequence[Dict[str, Any]],
                           current_context: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    device = _resolve_device(current_context.get("device"))
    paper_dir = Path(output_dir) / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    figure_builders = [
        ("paper_fig_01_teaser.png", _build_teaser_figure),
        ("paper_fig_02_projection_setups.png", _build_projection_setup_figure),
        ("paper_fig_03_pipeline.png", _build_pipeline_figure),
        ("paper_fig_04_ablation.png", _build_ablation_figure),
        ("paper_fig_05_projection_comparison.png", _build_projection_comparison_figure),
        ("paper_fig_06_prior_comparison.png", _build_prior_comparison_figure),
        ("paper_fig_07_voxel_gallery.png", _build_voxel_gallery_figure),
        ("paper_fig_08_sketch_gallery.png", _build_sketch_gallery_figure),
        ("paper_fig_09_printable_gallery.png", _build_printable_gallery_figure),
    ]

    for filename, builder in figure_builders:
        fig_path = paper_dir / filename
        entry = {"file": filename, "status": "placeholder", "sources": [], "notes": ""}
        try:
            if builder is _build_teaser_figure:
                builder(current_summary, all_summaries, fig_path, entry, device)
            elif builder is _build_projection_setup_figure:
                builder(current_summary, all_summaries, fig_path, entry, device)
            elif builder is _build_pipeline_figure:
                builder(current_summary, fig_path, entry, device)
            elif builder is _build_ablation_figure:
                builder(current_summary, current_context, fig_path, entry, device)
            elif builder is _build_projection_comparison_figure:
                builder(current_summary, all_summaries, fig_path, entry, device)
            elif builder is _build_prior_comparison_figure:
                builder(current_summary, all_summaries, fig_path, entry, device)
            elif builder is _build_voxel_gallery_figure:
                builder(current_summary, fig_path, entry, device)
            elif builder is _build_sketch_gallery_figure:
                builder(current_summary, fig_path, entry, device)
            elif builder is _build_printable_gallery_figure:
                builder(current_summary, fig_path, entry, device)
        except Exception as exc:
            placeholder = _placeholder_image(filename, f"Figure generation failed: {exc}")
            _save_paper_figure(fig_path, placeholder)
            entry.update({
                "status": "placeholder",
                "sources": entry.get("sources", []),
                "notes": f"Generation failed and fell back to placeholder: {exc}",
            })
            traceback.print_exc()
        manifest.append(entry)

    manifest_lines = []
    for entry in manifest:
        sources = ", ".join(entry.get("sources", [])) if entry.get("sources") else "N/A"
        manifest_lines.append(f"{entry['file']} | {entry['status']} | sources: {sources}")
        manifest_lines.append(f"  notes: {entry.get('notes', '')}")
    manifest_path = paper_dir / "manifest.txt"
    _save_text(manifest_path, "\n".join(manifest_lines) + "\n")

    index_payload = {
        "current_run": current_summary["run_id"],
        "figures": manifest,
    }
    index_path = paper_dir / "index.json"
    _json_dump(index_path, index_payload)
    return {
        "paper_dir": str(paper_dir),
        "manifest": str(manifest_path),
        "index": str(index_path),
        "figures": manifest,
    }


def generate_post_export_bundle(context: Dict[str, Any]) -> Dict[str, Any]:
    import time

    bundle_start_time = time.time()
    output_dir = str(Path(context["output_dir"]).resolve())
    sample_id = int(context.get("sample_id", 1))
    current_summary = _normalize_run_summary(Path(output_dir))
    if current_summary is None:
        raise RuntimeError(f"Unable to normalize current run summary from {output_dir}")

    current_summary["artifact_paths"].update({
        "obj": current_summary["artifact_paths"].get("obj") or context.get("obj_path"),
        "voxels": current_summary["artifact_paths"].get("voxels") or context.get("voxels_path"),
    })
    current_summary["renderer_metrics"] = context.get("renderer_metrics", current_summary.get("renderer_metrics", {}))
    current_summary["aggregate_metrics"] = {
        **current_summary["aggregate_metrics"],
        **{k: v for k, v in (context.get("export_metrics") or {}).items() if k in {"mean_iou", "mean_dice", "occupied_voxels"}},
    }
    _backfill_normalized_losses(current_summary)
    _backfill_normal_consistency(current_summary, _resolve_device(context.get("device")))

    metrics_report_paths = write_metrics_report(context, current_summary)
    current_summary["artifact_paths"]["metrics_report"] = metrics_report_paths["sample_report"]

    chart_paths = generate_analytic_charts(current_summary["metrics_history"], output_dir, sample_id)

    visuals = generate_current_sample_visuals(context, current_summary)
    current_summary["artifact_paths"].update(visuals)

    device = _resolve_device(context.get("device"))
    mesh = _load_display_mesh(current_summary, device)
    rotation_path = Path(output_dir) / f"sample_{sample_id}_rotation.gif"
    rotation_gif = _render_turntable(
        mesh,
        device=device,
        frames=int(context.get("rotation_frames", 72)),
        fps=int(context.get("rotation_fps", 18)),
        camera_mode=current_summary.get("camera_mode", "ortho"),
        out_path=rotation_path,
    )
    if rotation_gif:
        current_summary["artifact_paths"]["rotation_gif"] = rotation_gif

    all_summaries = scan_history_runs(output_dir, context.get("history_roots"))
    all_summaries = [s for s in all_summaries if s["source_dir"] != current_summary["source_dir"]]
    all_summaries = [current_summary] + all_summaries

    paper_bundle = generate_paper_figures(current_summary, all_summaries, context, output_dir)
    context["post_export_seconds"] = time.time() - bundle_start_time
    metrics_report_paths = write_metrics_report(context, current_summary)
    bundle_index = {
        "run_id": current_summary["run_id"],
        "output_dir": output_dir,
        "metrics_report": metrics_report_paths,
        "charts": chart_paths,
        "visuals": visuals,
        "rotation_gif": rotation_gif,
        "paper_figures": paper_bundle,
    }
    _json_dump(Path(output_dir) / "post_export_bundle.json", bundle_index)
    return bundle_index
