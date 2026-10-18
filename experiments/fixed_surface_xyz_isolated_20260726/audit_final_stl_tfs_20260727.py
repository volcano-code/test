#!/usr/bin/env python3
"""Audit a delivered STL with the fixed TFS cameras used by the soft stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
GENERATION = ROOT / "generation"
if str(GENERATION) not in sys.path:
    sys.path.insert(0, str(GENERATION))

from post_export import _build_mesh_from_verts_faces, _render_mesh_rgb_and_masks  # noqa: E402
from utils import get_tfs_view_specs  # noqa: E402


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred_fg = pred > 127
    target_fg = target > 127
    intersection = int(np.logical_and(pred_fg, target_fg).sum())
    union = int(np.logical_or(pred_fg, target_fg).sum())
    denom = int(pred_fg.sum() + target_fg.sum())
    iou = intersection / union if union else 1.0
    dice = 2.0 * intersection / denom if denom else 1.0
    return float(iou), float(dice)


def d4(image: np.ndarray, op_id: int) -> np.ndarray:
    out = np.rot90(image, int(op_id) % 4)
    if int(op_id) >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--input-views", type=int, default=3)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--mesh-extent",
        type=float,
        default=1.4,
        help="Normalized largest mesh extent. TFS targets use a 70%% image-frame extent.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    mesh_tm = trimesh.load(args.stl, force="mesh", process=True)
    mesh_pt = _build_mesh_from_verts_faces(
        np.asarray(mesh_tm.vertices),
        np.asarray(mesh_tm.faces),
        device,
        color=(0.22, 0.68, 0.64),
    )
    mesh_pt.scale_verts_(float(args.mesh_extent) / 1.65)
    specs = get_tfs_view_specs(args.input_views, args.mirror)
    if len(specs) != len(args.targets):
        raise ValueError(f"{len(specs)} cameras but {len(args.targets)} targets")
    angles = [(float(elev), float(azim)) for _, elev, azim in specs]
    rgbs = []
    masks = []
    for angle in angles:
        view_rgbs, view_masks = _render_mesh_rgb_and_masks(
            mesh_pt,
            [angle],
            device=device,
            image_size=args.image_size,
            camera_mode="ortho",
        )
        rgbs.extend(view_rgbs)
        masks.extend(view_masks)

    rows = []
    fig, axes = plt.subplots(len(specs), 4, figsize=(8.4, 2.05 * len(specs)))
    for idx, ((label, elev, azim), target_path, rgb, mask) in enumerate(
        zip(specs, args.targets, rgbs, masks)
    ):
        target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(target_path)
        target = cv2.resize(
            target,
            (args.image_size, args.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        direct_iou, direct_dice = binary_metrics(mask, target)
        # The delivered STL uses canonical XYZ, whereas the trainer stores ZYX
        # with flipped first and third axes. These fixed image transforms are
        # therefore protocol conventions, not target-dependent registration.
        protocol_op = 2 if label in {"top", "down"} else 4
        protocol_mask = d4(mask, protocol_op)
        protocol_iou, protocol_dice = binary_metrics(protocol_mask, target)
        best_op = 0
        best_iou = direct_iou
        best_dice = direct_dice
        best_mask = mask
        for op_id in range(1, 8):
            candidate = d4(mask, op_id)
            iou, dice = binary_metrics(candidate, target)
            if iou > best_iou:
                best_op = op_id
                best_iou = iou
                best_dice = dice
                best_mask = candidate

        overlay = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
        overlay[..., 1] = (target > 127).astype(np.uint8) * 255
        overlay[..., 0] = (protocol_mask > 127).astype(np.uint8) * 255
        rows.append(
            {
                "index": idx,
                "label": label,
                "elevation": float(elev),
                "azimuth": float(azim),
                "target": str(Path(target_path).resolve()),
                "direct_iou": direct_iou,
                "direct_dice": direct_dice,
                "protocol_frame_d4_op": protocol_op,
                "protocol_iou": protocol_iou,
                "protocol_dice": protocol_dice,
                "diagnostic_best_d4_op": best_op,
                "diagnostic_best_d4_iou": best_iou,
                "diagnostic_best_d4_dice": best_dice,
            }
        )

        axes[idx, 0].imshow(target, cmap="gray", vmin=0, vmax=255)
        axes[idx, 0].set_title(f"{label}: target")
        axes[idx, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[idx, 1].set_title(f"direct IoU {direct_iou:.3f}")
        axes[idx, 2].imshow(protocol_mask, cmap="gray", vmin=0, vmax=255)
        axes[idx, 2].set_title(f"protocol IoU {protocol_iou:.3f}")
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title("red=mesh, green=target")
        for axis in axes[idx]:
            axis.axis("off")

        cv2.imwrite(str(output / f"view_{idx}_{label}_target.png"), target)
        cv2.imwrite(str(output / f"view_{idx}_{label}_mesh_direct.png"), mask)
        cv2.imwrite(str(output / f"view_{idx}_{label}_mesh_protocol.png"), protocol_mask)
        cv2.imwrite(str(output / f"view_{idx}_{label}_mesh_d4_diagnostic.png"), best_mask)
        cv2.imwrite(str(output / f"view_{idx}_{label}_render.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    fig.tight_layout()
    fig.savefig(output / "final_stl_tfs_audit.png", dpi=220, facecolor="white")
    plt.close(fig)

    report = {
        "stl": str(Path(args.stl).resolve()),
        "note": "D4-aligned values are orientation diagnostics, not protocol metrics.",
        "mesh": {
            "vertices": int(len(mesh_tm.vertices)),
            "faces": int(len(mesh_tm.faces)),
            "body_count": int(mesh_tm.body_count),
            "watertight_after_vertex_welding": bool(mesh_tm.is_watertight),
            "winding_consistent": bool(mesh_tm.is_winding_consistent),
            "extents": [float(v) for v in mesh_tm.extents],
        },
        "direct_mean_iou": float(np.mean([row["direct_iou"] for row in rows])),
        "direct_min_iou": float(np.min([row["direct_iou"] for row in rows])),
        "direct_mean_dice": float(np.mean([row["direct_dice"] for row in rows])),
        "protocol_mean_iou": float(np.mean([row["protocol_iou"] for row in rows])),
        "protocol_min_iou": float(np.min([row["protocol_iou"] for row in rows])),
        "protocol_mean_dice": float(np.mean([row["protocol_dice"] for row in rows])),
        "diagnostic_d4_mean_iou": float(np.mean([row["diagnostic_best_d4_iou"] for row in rows])),
        "views": rows,
    }
    with (output / "REPORT.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
