#!/usr/bin/env python3
"""Export Sadekar mesh/voxel projections in canonical XYZ order.

Sadekar's official TFS camera order is front(Y), side(X), top(Z).  The
benchmark manifest therefore supplies targets in Y/X/Z order.  This script
keeps raw artifacts intact and exports both raw-TFS and canonical-XYZ views.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
DEFAULT_MANIFEST = (
    ROOT
    / "results/fixed3_mr1_connected10_pending_20260722"
    / "benchmark_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/fig4_connected06_sadekar_projection_audit_20260728"
)
SOURCES = {
    "mesh": Path(
        "/data/xvxinyao/ShadowArt-Revisited/mesh_results/"
        "fixed3_mr1_connected10_sadekar_20260722/mesh/"
        "connected_06/seed_17"
    ),
    "voxel": Path(
        "/data/xvxinyao/ShadowArt-Revisited/voxel_results/"
        "fixed3_mr1_connected10_sadekar_20260722/voxel/"
        "connected_06/seed_17"
    ),
}

# Canonical X/Y/Z view -> Sadekar raw TFS index.
CANONICAL_TO_TFS = {"x": 1, "y": 0, "z": 2}
CANONICAL_MIRROR_TO_TFS = {"x": 4, "y": 3, "z": 5}


def load_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image)


def minority_mask(gray: np.ndarray) -> np.ndarray:
    bright = gray > 127
    return bright if bright.mean() <= 0.5 else ~bright


def best_mask(gray: np.ndarray, reference: np.ndarray) -> np.ndarray:
    bright = gray > 127
    candidates = (bright, ~bright)
    return max(candidates, key=lambda mask: iou(mask, reference))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(int(union), 1))


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def prediction_path(kind: str, source: Path, index: int) -> Path:
    if kind == "mesh":
        return source / f"sample_0_pred_view_{index}.png"
    return source / f"sample_0view_{index}_silh.png"


def target_path(source: Path, index: int) -> Path:
    return source / f"sample_0view_{index}.png"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def contact_sheet(
    kind: str,
    rows: dict[str, list[Path]],
    output: Path,
) -> None:
    tile = 256
    left = 150
    top = 46
    gap = 12
    width = left + 3 * tile + 2 * gap
    height = top + 3 * tile + 2 * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    row_font = font(18)
    for col, axis in enumerate(("X", "Y", "Z")):
        x = left + col * (tile + gap)
        draw.text((x + tile / 2, 8), axis, fill="#263238", font=title_font, anchor="ma")
    row_names = (
        ("declared_target", "Declared target"),
        ("author_target", "Author target"),
        ("prediction", f"Sadekar {kind}"),
    )
    for row_index, (key, label) in enumerate(row_names):
        y = top + row_index * (tile + gap)
        draw.text((left - 10, y + tile / 2), label, fill="#263238", font=row_font, anchor="rm")
        for col, path in enumerate(rows[key]):
            image = Image.open(path).convert("RGB")
            image = image.resize((tile, tile), Image.Resampling.NEAREST)
            x = left + col * (tile + gap)
            canvas.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", default="connected_06")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    case = next(row for row in payload["cases"] if row["case_id"] == args.case)
    axes = ("x", "y", "z")
    declared_paths = {
        axis: Path(path) for axis, path in zip(axes, case["target_paths_xyz"])
    }
    args.output.mkdir(parents=True, exist_ok=True)

    declared_masks: dict[str, np.ndarray] = {}
    for axis in axes:
        raw = load_gray(declared_paths[axis])
        declared_masks[axis] = minority_mask(raw)
        save_mask(
            declared_masks[axis],
            args.output / "declared_targets_xyz" / f"target_{axis}.png",
        )

    audit: dict[str, object] = {
        "case_id": args.case,
        "protocol": payload["protocol"],
        "sadekar_raw_tfs_order": [
            "front_y",
            "side_x",
            "top_z",
            "back_y",
            "opposite_side_x",
            "bottom_z",
        ],
        "canonical_xyz_to_raw_tfs": CANONICAL_TO_TFS,
        "canonical_mirror_xyz_to_raw_tfs": CANONICAL_MIRROR_TO_TFS,
        "methods": {},
    }

    method_targets: dict[str, dict[str, np.ndarray]] = {}
    for kind, source in SOURCES.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        method_out = args.output / f"sadekar_{kind}"
        raw_out = method_out / "raw_tfs"
        raw_out.mkdir(parents=True, exist_ok=True)

        for index in range(6):
            shutil.copy2(target_path(source, index), raw_out / f"target_view_{index}.png")
            shutil.copy2(
                prediction_path(kind, source, index),
                raw_out / f"prediction_view_{index}.png",
            )

        mapping_matrix: dict[str, dict[str, float]] = {}
        for axis in axes:
            reference = declared_masks[axis]
            mapping_matrix[axis] = {}
            for index in range(3):
                gray = load_gray(target_path(source, index), size=(reference.shape[1], reference.shape[0]))
                mapping_matrix[axis][f"tfs_view_{index}"] = iou(
                    best_mask(gray, reference), reference
                )

        canonical_rows = {
            "declared_target": [],
            "author_target": [],
            "prediction": [],
        }
        method_targets[kind] = {}
        prediction_scores: dict[str, float] = {}
        for axis in axes:
            index = CANONICAL_TO_TFS[axis]
            reference = declared_masks[axis]
            size = (reference.shape[1], reference.shape[0])
            author_mask = best_mask(load_gray(target_path(source, index), size=size), reference)
            prediction_mask = best_mask(
                load_gray(prediction_path(kind, source, index), size=size),
                reference,
            )
            method_targets[kind][axis] = author_mask
            prediction_scores[axis] = iou(prediction_mask, reference)

            declared_out = method_out / "canonical_xyz" / f"target_declared_{axis}.png"
            author_out = method_out / "canonical_xyz" / f"target_author_{axis}.png"
            prediction_out = method_out / "canonical_xyz" / f"prediction_{axis}.png"
            save_mask(reference, declared_out)
            save_mask(author_mask, author_out)
            save_mask(prediction_mask, prediction_out)
            canonical_rows["declared_target"].append(declared_out)
            canonical_rows["author_target"].append(author_out)
            canonical_rows["prediction"].append(prediction_out)

            mirror_index = CANONICAL_MIRROR_TO_TFS[axis]
            mirror_reference = np.fliplr(reference)
            mirror_author = best_mask(
                load_gray(target_path(source, mirror_index), size=size),
                mirror_reference,
            )
            mirror_prediction = best_mask(
                load_gray(prediction_path(kind, source, mirror_index), size=size),
                mirror_reference,
            )
            save_mask(
                mirror_author,
                method_out / "canonical_mirrored_xyz" / f"target_author_neg_{axis}.png",
            )
            save_mask(
                mirror_prediction,
                method_out / "canonical_mirrored_xyz" / f"prediction_neg_{axis}.png",
            )

        contact_sheet(
            kind,
            canonical_rows,
            args.output / f"sadekar_{kind}_targets_and_projections_xyz.png",
        )
        audit["methods"][kind] = {
            "source": str(source),
            "mapping_iou_matrix": mapping_matrix,
            "prediction_iou_xyz": prediction_scores,
            "prediction_mean_iou": float(np.mean(list(prediction_scores.values()))),
        }

    cross_method_target_iou = {
        axis: iou(method_targets["mesh"][axis], method_targets["voxel"][axis])
        for axis in axes
    }
    audit["mesh_voxel_author_target_iou_xyz"] = cross_method_target_iou
    audit_path = args.output / "axis_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))

    print(
        json.dumps(
            {
                "output": str(args.output),
                "axis_audit": str(audit_path),
                "canonical_mapping": CANONICAL_TO_TFS,
                "mesh_voxel_target_iou": cross_method_target_iou,
                "prediction_mean_iou": {
                    kind: audit["methods"][kind]["prediction_mean_iou"]
                    for kind in ("mesh", "voxel")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
