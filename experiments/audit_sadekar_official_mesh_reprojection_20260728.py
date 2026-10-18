#!/usr/bin/env python3
"""Re-render the archived Sadekar mesh with its original TFS cameras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import FoVOrthographicCameras
from pytorch3d.structures import Meshes


SAD_MESH_ROOT = Path("/data/xvxinyao/ShadowArt-Revisited/mesh")
if str(SAD_MESH_ROOT) not in sys.path:
    sys.path.insert(0, str(SAD_MESH_ROOT))

from utills import create_cameras_TFS_mode, get_silhouette_renderer  # noqa: E402


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first > 0
    second_bool = second > 0
    union = np.logical_or(first_bool, second_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_bool, second_bool).sum() / union)


def centroid(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zdist", type=float, default=1.7)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    obj_path = args.result_dir / "sample_0_output.obj"
    if not obj_path.is_file():
        raise FileNotFoundError(obj_path)

    vertices, faces, _ = load_obj(str(obj_path), load_textures=False)
    mesh = Meshes(
        verts=[vertices.to(device)],
        faces=[faces.verts_idx.to(device)],
    )
    _, rotations, translations = create_cameras_TFS_mode(
        device,
        args.zdist,
        True,
        "ortho",
    )
    renderer = get_silhouette_renderer(
        device,
        FoVOrthographicCameras(device=device),
        args.resolution,
    )

    rows: list[dict[str, object]] = []
    for index in range(6):
        rendered = renderer(
            mesh,
            R=rotations[index].unsqueeze(0).to(device),
            T=translations[index].unsqueeze(0).to(device),
        )
        alpha = rendered[..., 3][0].detach().cpu().numpy()
        alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        _, binary = cv2.threshold(
            alpha_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        reference_path = args.result_dir / f"sample_0_pred_view_{index}.png"
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise FileNotFoundError(reference_path)
        _, reference = cv2.threshold(reference, 0, 255, cv2.THRESH_BINARY)

        output_path = args.output / f"rerender_view_{index}.png"
        cv2.imwrite(str(output_path), binary)
        difference = cv2.absdiff(binary, reference)
        cv2.imwrite(str(args.output / f"difference_view_{index}.png"), difference)

        rows.append(
            {
                "view": index,
                "iou_to_archived_prediction": binary_iou(binary, reference),
                "different_pixels": int(np.count_nonzero(difference)),
                "rerender_foreground_pixels": int(np.count_nonzero(binary)),
                "archived_foreground_pixels": int(np.count_nonzero(reference)),
                "rerender_centroid_xy": centroid(binary),
                "archived_centroid_xy": centroid(reference),
                "rerender_path": str(output_path),
                "archived_path": str(reference_path),
            }
        )

    report = {
        "source_obj": str(obj_path),
        "camera": {
            "protocol": "Sadekar TFS",
            "projection": "FoVOrthographicCameras",
            "mirror_mode": True,
            "zdist": args.zdist,
            "resolution": args.resolution,
        },
        "views": rows,
        "mean_iou_to_archived_prediction": float(
            np.mean([row["iou_to_archived_prediction"] for row in rows])
        ),
        "max_different_pixels": int(max(row["different_pixels"] for row in rows)),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
