#!/usr/bin/env python3
"""Controlled soft-to-mesh conversion comparison for Figure 4.

Every method starts from the same frozen 256^3 probability field and target
pair.  The comparison isolates surface conversion from soft optimization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, generate_binary_structure, label
from skimage.measure import marching_cubes
from pytorch3d.ops import cubify


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "shadowforge_pipeline")]

from shadowforge_pipeline.unified_run import mesh_qa, n2, tw


DEFAULT_SOURCE = ROOT / "results/geometry_module_ablations_matched_v1_20260716"
DEFAULT_OUTPUT = ROOT / "results/fig4_soft_to_mesh_conversion_20260717"
S6 = generate_binary_structure(3, 1)


def load_mask(path: Path, resolution: int = 256) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def projections(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = tw.wall_axes_to_image(tw.tw.project_left_wall_x(volume))
    right = tw.wall_axes_to_image(tw.tw.project_right_wall_y(volume))
    return left.astype(bool), right.astype(bool)


def overlap(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    denom = int(pred.sum() + target.sum())
    return inter / max(union, 1), 2.0 * inter / max(denom, 1)


def direct_mc(volume: np.ndarray) -> trimesh.Trimesh:
    verts, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), level=0.5)
    verts -= 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True, validate=True)


def direct_cubify(volume: np.ndarray) -> trimesh.Trimesh:
    """Native PyTorch3D cube-surface extraction, with no cleanup or smoothing."""
    tensor = torch.from_numpy(volume.astype(np.float32))[None]
    mesh = cubify(tensor, thresh=0.5)
    return trimesh.Trimesh(
        vertices=mesh.verts_packed().cpu().numpy(),
        faces=mesh.faces_packed().cpu().numpy(),
        process=False,
        validate=False,
    )


def sdf_only_volume(raw: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a binary signed-distance field without topology repair."""
    inside = distance_transform_edt(raw)
    outside = distance_transform_edt(~raw)
    sdf = inside - outside
    return gaussian_filter(sdf.astype(np.float32), sigma=sigma) >= 0.0


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    parts = mesh.split(only_watertight=False)
    return {
        "watertight": bool(mesh.is_watertight and mesh.is_winding_consistent),
        "mesh_parts": int(len(parts)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }


def render_mesh(mesh: trimesh.Trimesh, output: Path, color: str) -> None:
    fig = plt.figure(figsize=(4.0, 4.0), dpi=240)
    ax = fig.add_subplot(111, projection="3d")
    faces = mesh.faces
    stride = max(1, len(faces) // 180_000)
    sampled = faces[::stride]
    vertices = mesh.vertices
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=sampled, color=color, linewidth=0.0, antialiased=False,
        shade=True,
    )
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    radius = max(float(np.ptp(bounds, axis=0).max()) * 0.56, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)


def save_projection_panel(
    targets: tuple[np.ndarray, np.ndarray],
    preds: tuple[np.ndarray, np.ndarray],
    output: Path,
) -> None:
    tile = 256
    canvas = Image.new("RGB", (tile * 2, tile * 2 + 44), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("Target X", "Target Y", "Rendered X", "Rendered Y")
    arrays = (*targets, *preds)
    positions = ((0, 22), (tile, 22), (0, tile + 44), (tile, tile + 44))
    for label_text, array, (x, y) in zip(labels, arrays, positions):
        tile_image = Image.fromarray(np.where(array, 255, 0).astype(np.uint8), mode="L").convert("RGB")
        canvas.paste(tile_image, (x, y))
        draw.text((x + 6, y - 18), label_text, fill="black")
    canvas.save(output)


def evaluate_volume(
    volume: np.ndarray,
    targets: tuple[np.ndarray, np.ndarray],
    mesh: trimesh.Trimesh,
) -> dict:
    preds = projections(volume)
    pairs = [overlap(pred, target) for pred, target in zip(preds, targets)]
    _, components = label(volume, structure=S6)
    return {
        "occupied_voxels": int(volume.sum()),
        "mean_iou": float(np.mean([pair[0] for pair in pairs])),
        "mean_dice": float(np.mean([pair[1] for pair in pairs])),
        "view_ious": [float(pair[0]) for pair in pairs],
        "view_dices": [float(pair[1]) for pair in pairs],
        "components6": int(components),
        "single_body": bool(components == 1),
        **mesh_stats(mesh),
    }


def mean_iou(volume: np.ndarray, targets: tuple[np.ndarray, np.ndarray]) -> float:
    preds = projections(volume)
    return float(np.mean([overlap(pred, target)[0] for pred, target in zip(preds, targets)]))


def run_case(
    case_dir: Path,
    output: Path,
    thresholds: tuple[float, ...],
    sdf_sigma: float,
    direct_threshold_override: float | None = None,
) -> dict:
    manifest = json.loads((case_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    # Artifacts produced before the 2026-07-18 canonical-frame fix were saved
    # with both projected image axes reversed.  Apply the one frozen global
    # mapping used by the fixed two-wall protocol; never optimize it per case.
    probability = np.flip(
        np.load(case_dir / "soft_probability_256.npy").astype(np.float32),
        axis=(0, 2),
    )
    targets = (
        load_mask(case_dir / "inputs/left_wall_x_target.png"),
        load_mask(case_dir / "inputs/right_wall_y_target.png"),
    )
    shadowforge_volume = np.load(case_dir / "selected_display_voxels256.npy").astype(bool)

    raw_candidates = [(threshold, probability >= threshold) for threshold in thresholds]
    # The default is the projection-best threshold.  For a controlled
    # projection-only sparse-seed demonstration, an explicitly declared fixed
    # threshold may instead be supplied.  This is deliberately not tuned per
    # object: every method still uses the same frozen field and evaluator.
    if direct_threshold_override is None:
        direct_threshold, raw = max(raw_candidates, key=lambda item: mean_iou(item[1], targets))
    else:
        direct_threshold = float(direct_threshold_override)
        raw = probability >= direct_threshold
    sdf_candidates = [
        (threshold, sdf_only_volume(candidate, sdf_sigma))
        for threshold, candidate in raw_candidates
    ]
    sdf_threshold, sdf_volume = max(sdf_candidates, key=lambda item: mean_iou(item[1], targets))

    methods = {
        "direct_cubify": (raw, "#7C6AB0", direct_threshold, direct_cubify),
        "direct_marching_cubes": (raw, "#90A4AE", direct_threshold, direct_mc),
        "sdf_only": (sdf_volume, "#80CBC4", sdf_threshold, direct_mc),
        "shadowforge_conversion": (shadowforge_volume, "#FFB74D", None, direct_mc),
    }
    case_output = output / case_dir.parent.name
    case_output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, (volume, color, selected_threshold, extractor) in methods.items():
        method_dir = case_output / method
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "volume.npy", volume.astype(np.uint8))
        mesh = extractor(volume)
        mesh.export(method_dir / "mesh.stl")
        render_mesh(mesh, method_dir / "aligned_3d.png", color)
        preds = projections(volume)
        save_projection_panel(targets, preds, method_dir / "target_vs_projection.png")
        row = {
            "task": case_dir.parent.name,
            "method": method,
            "selected_threshold": selected_threshold,
            **evaluate_volume(volume, targets, mesh),
        }
        (method_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)

    shutil.copy2(case_dir / "soft_probability_256.npy", case_output / "frozen_soft_probability_256.npy")
    return {
        "task": case_dir.parent.name,
        "source": str(case_dir),
        "threshold_grid": list(thresholds),
        "sdf_sigma": sdf_sigma,
        "selected_candidate": manifest.get("selected", {}).get("candidate"),
        "rows": rows,
    }


def aggregate(records: list[dict]) -> list[dict]:
    methods = [row["method"] for row in records[0]["rows"]]
    summary = []
    for method in methods:
        rows = [next(row for row in record["rows"] if row["method"] == method) for record in records]
        summary.append({
            "method": method,
            "n_tasks": len(rows),
            "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
            "std_iou": float(np.std([row["mean_iou"] for row in rows])),
            "mean_dice": float(np.mean([row["mean_dice"] for row in rows])),
            "std_dice": float(np.std([row["mean_dice"] for row in rows])),
            "single_body_rate": float(np.mean([row["single_body"] for row in rows])),
            "watertight_rate": float(np.mean([row["watertight"] for row in rows])),
            "mean_mesh_parts": float(np.mean([row["mesh_parts"] for row in rows])),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="*", default=["sword_fox", "future_mech", "parrot_snail", "owl_turtle"])
    parser.add_argument("--thresholds", default="0.15,0.25,0.35,0.45,0.55")
    parser.add_argument("--sdf-sigma", type=float, default=0.5)
    parser.add_argument(
        "--direct-threshold-override",
        type=float,
        default=None,
        help="Declared fixed threshold for a projection-only sparse-seed baseline.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(float(item) for item in args.thresholds.split(",") if item.strip())
    records = []
    for task in args.tasks:
        case_dir = args.source / task / "full"
        print(json.dumps({"stage": "conversion_case", "task": task}), flush=True)
        records.append(
            run_case(
                case_dir,
                args.output,
                thresholds,
                args.sdf_sigma,
                args.direct_threshold_override,
            )
        )
    summary = aggregate(records)
    payload = {
        "schema": "fig4_soft_to_mesh_conversion_v1",
        "controls": {
            "same_probability_field": True,
            "threshold_grid": list(thresholds),
            "same_targets": True,
            "same_projection_evaluator": True,
            "same_mesh_qa": True,
            "cubify_operator": "pytorch3d.ops.cubify(thresh=0.5); no cleanup or smoothing",
            "sdf_only_sigma": args.sdf_sigma,
            "direct_threshold_override": args.direct_threshold_override,
        },
        "records": records,
        "aggregate": summary,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    columns = list(summary[0])
    with (args.output / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in summary:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
