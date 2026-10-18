#!/usr/bin/env python3
"""Screen downloaded anime silhouettes for ShadowForge-friendly structure.

The score is a pre-generation filter only.  It favors silhouettes that are
single connected, thick, compact enough for positive 3D sculpture generation,
and not dominated by hair strands, long weapons, or tiny isolated details.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SEARCH_ROOTS = [
    Path("/home/lin/下载"),
    Path("/home/lin/xvxinyao"),
]
OUT = ROOT / "results/anime_structure_friendly_screen_20260709"
SIZE = 256


def iter_images() -> list[Path]:
    keys = ("anime", "silhouette", "character", "original_anime", "latest_30", "single_connected")
    paths: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            text = str(path).lower()
            if any(k in text for k in keys):
                paths.append(path)
    return sorted(set(paths))


def load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    # Accept both black-on-white and white-on-black.  The foreground is the
    # smaller extreme component unless the file is already white-on-black.
    dark = arr < 128
    light = arr > 127
    border_dark = np.r_[dark[0], dark[-1], dark[:, 0], dark[:, -1]].mean()
    border_light = np.r_[light[0], light[-1], light[:, 0], light[:, -1]].mean()
    if border_dark > border_light:
        mask = light
    else:
        mask = dark
    mask = binary_closing(mask, iterations=1)
    mask = binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    lab, n = label(mask)
    if n == 0:
        return mask & False, 0, 0.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    idx = int(counts.argmax())
    largest = lab == idx
    ratio = float(counts[idx] / max(int(mask.sum()), 1))
    return largest, int(n), ratio


def metrics(path: Path) -> dict:
    mask = load_mask(path)
    largest, components, largest_ratio = largest_component(mask)
    area = float(largest.mean())
    ys, xs = np.where(largest)
    if len(xs) == 0:
        return {"path": str(path), "valid": False, "reason": "empty"}
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    bbox_area = max(w * h, 1)
    bbox_fill = float(largest.sum() / bbox_area)
    aspect = float(h / max(w, 1))
    dist = distance_transform_edt(largest)
    thin_ratio = float((largest & (dist < 2.0)).sum() / max(int(largest.sum()), 1))
    robust_core = float((dist >= 3.0).sum() / max(int(largest.sum()), 1))
    contours, hierarchy = cv2.findContours(largest.astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    complexity = float(perimeter / max(np.sqrt(float(largest.sum())), 1.0))
    holes = 0
    if hierarchy is not None:
        holes = int(sum(1 for hrow in hierarchy[0] if hrow[3] >= 0))
    bottom = largest[int(SIZE * 0.86) :, :]
    foot_contact = float(bottom.sum() / max(int(largest.sum()), 1))

    penalties = 0.0
    penalties += max(0.0, 0.10 - area) * 20
    penalties += max(0.0, area - 0.42) * 12
    penalties += max(0.0, components - 1) * 2.0
    penalties += max(0.0, 0.96 - largest_ratio) * 10
    penalties += max(0.0, thin_ratio - 0.42) * 8
    penalties += max(0.0, 0.18 - robust_core) * 8
    penalties += max(0.0, complexity - 18.0) * 0.35
    penalties += max(0.0, holes - 4) * 0.8
    penalties += max(0.0, 1.1 - aspect) * 1.5
    penalties += max(0.0, aspect - 3.2) * 1.5
    penalties += max(0.0, 0.015 - foot_contact) * 20
    score = 100.0 - penalties
    pass_basic = (
        area >= 0.10
        and area <= 0.42
        and components <= 2
        and largest_ratio >= 0.96
        and thin_ratio <= 0.48
        and robust_core >= 0.14
        and complexity <= 24.0
        and holes <= 8
        and 0.9 <= aspect <= 3.5
    )
    return {
        "path": str(path),
        "name": path.stem,
        "valid": True,
        "pass_basic": bool(pass_basic),
        "score": round(float(score), 4),
        "area": round(area, 6),
        "bbox_fill": round(bbox_fill, 6),
        "aspect_h_over_w": round(aspect, 6),
        "components": components,
        "largest_ratio": round(largest_ratio, 6),
        "thin_ratio": round(thin_ratio, 6),
        "robust_core_ratio": round(robust_core, 6),
        "perimeter_complexity": round(complexity, 6),
        "holes": holes,
        "foot_contact": round(foot_contact, 6),
    }


def save_normalized(path: Path, dst: Path) -> None:
    mask = load_mask(path)
    img = Image.fromarray(mask.astype(np.uint8) * 255).resize((512, 512), Image.Resampling.NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def make_contact_sheet(rows: list[dict], out: Path, n: int = 40) -> None:
    rows = rows[:n]
    cell = 112
    cols = 8
    label_h = 34
    w = cols * cell
    h = ((len(rows) + cols - 1) // cols) * (cell + label_h)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x = (i % cols) * cell
        y = (i // cols) * (cell + label_h)
        mask = load_mask(Path(row["path"]))
        img = Image.fromarray(mask.astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(img.convert("RGB"), (x, y))
        draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(130, 130, 130))
        draw.text((x + 2, y + cell + 2), f"{i+1:02d} S={row['score']:.1f}", fill=(0, 0, 0))
    canvas.save(out)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "selected").mkdir(parents=True)
    all_rows = [metrics(p) for p in iter_images()]
    valid = [r for r in all_rows if r.get("valid")]
    valid.sort(key=lambda r: (not r["pass_basic"], -float(r["score"]), float(r["thin_ratio"])))
    selected = [r for r in valid if r["pass_basic"] and r["score"] >= 85.0][:60]
    for idx, row in enumerate(selected, 1):
        save_normalized(Path(row["path"]), OUT / "selected" / f"{idx:03d}_{Path(row['path']).stem}.png")
    make_contact_sheet(selected, OUT / "selected_contact_sheet.png", n=min(40, len(selected)))
    result = {
        "schema": "anime_structure_friendly_screen_20260709",
        "total_images": len(all_rows),
        "valid_images": len(valid),
        "selected_count": len(selected),
        "criteria": {
            "area": "0.10..0.42",
            "components": "<=2",
            "largest_ratio": ">=0.96",
            "thin_ratio": "<=0.48",
            "robust_core_ratio": ">=0.14",
            "perimeter_complexity": "<=24",
            "holes": "<=8",
            "aspect_h_over_w": "0.9..3.5",
            "score": ">=85 for selected",
        },
        "selected_dir": str(OUT / "selected"),
        "contact_sheet": str(OUT / "selected_contact_sheet.png"),
        "selected": selected,
        "top_all": valid[:100],
    }
    (OUT / "screen_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "total_images": len(all_rows),
        "valid_images": len(valid),
        "selected_count": len(selected),
        "selected_dir": str(OUT / "selected"),
        "contact_sheet": str(OUT / "selected_contact_sheet.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
