#!/usr/bin/env python3
"""Export input and model-projection panels for successful print trials."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
TRIALS = ROOT / "results/single_connected_unique_print_trials_20260708"
AXES = (2, 0, 1)


def read_mask(path: Path, size: int = 256) -> Image.Image:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = np.array(img)
    mask = arr < 128 if arr.mean() > 127 else arr > 127
    return Image.fromarray((mask.astype(np.uint8) * 255), "L")


def projection(volume: np.ndarray, axis: int) -> Image.Image:
    arr = volume.astype(bool).max(axis=axis).T
    return Image.fromarray((arr.astype(np.uint8) * 255), "L").resize((256, 256), Image.Resampling.NEAREST)


def six_from_three(imgs: list[Image.Image]) -> list[Image.Image]:
    return imgs + [ImageOps.mirror(im) for im in imgs]


def save_panel(imgs: list[Image.Image], out: Path, labels: list[str]) -> None:
    cell = 192
    pad = 16
    label_h = 24
    canvas = Image.new("RGB", (3 * cell + 4 * pad, 2 * (cell + label_h) + 3 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(imgs):
        r, c = divmod(i, 3)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        rgb = ImageOps.invert(im.resize((cell, cell), Image.Resampling.NEAREST)).convert("RGB")
        # White background with black foreground, publication-style.
        canvas.paste(rgb, (x, y + label_h))
        draw.text((x, y), labels[i], fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    summary = []
    for manifest_path in sorted(TRIALS.glob("*/MANIFEST.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "PASS":
            continue
        case_dir = manifest_path.parent
        out_dir = case_dir / "projection_review"
        out_dir.mkdir(parents=True, exist_ok=True)
        vol = np.load(case_dir / "selected_delivery.npy").astype(bool)
        input3 = [read_mask(Path(item["path"])) for item in manifest["inputs"]]
        proj3 = [projection(vol, axis) for axis in AXES]
        input6 = six_from_three(input3)
        proj6 = six_from_three(proj3)
        labels = ["view0", "view1", "view2", "view0 mirror", "view1 mirror", "view2 mirror"]
        for i, im in enumerate(input6):
            im.save(out_dir / f"input_{i:02d}.png")
        for i, im in enumerate(proj6):
            im.save(out_dir / f"model_projection_{i:02d}.png")
        save_panel(input6, out_dir / "inputs_6_panel.png", labels)
        save_panel(proj6, out_dir / "model_projections_6_panel.png", labels)
        summary.append({
            "case": case_dir.name,
            "out_dir": str(out_dir),
            "mean_iou": manifest["delivery"]["mean_iou"],
            "saving": manifest["delivery"]["material_saved_vs_hull"],
        })
    (TRIALS / "projection_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
