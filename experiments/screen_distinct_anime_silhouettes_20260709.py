#!/usr/bin/env python3
"""Screen anime silhouettes for visually distinctive, printable candidates.

This pass is stricter than the earlier structure-only screen: it rejects
near-blob shapes and ranks silhouettes that retain clear head/limb/prop cues
while avoiding hair-strand or weapon-thin details that break fixed 10cm export.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, distance_transform_edt, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/distinct_anime_silhouette_screen_20260709"
SIZE = 256

SOURCE_DIRS = [
    Path("/home/lin/下载/all_silhouettes_single_connected_unique/silhouettes"),
    Path("/home/lin/下载/all_silhouettes_single_connected_unique (2)/silhouettes"),
    Path("/home/lin/下载/latest_30_original_anime_silhouettes/latest_30_silhouettes"),
    Path("/home/lin/下载/original_anime_silhouettes_30 (4)"),
    Path("/home/lin/下载/original_anime_silhouettes_30(1)/original_anime_silhouettes_30"),
    Path("/home/lin/下载/original_anime_silhouettes_100"),
    Path("/home/lin/下载/original_anime_silhouettes_100 (2)"),
]

BAD_NAME_HINTS = {
    "grid",
    "shadowforge",
    "input",
    "projection",
    "birds",
    "owl",
    "parrot",
    "企鹅",
    "熊猫",
    "抱枕",
    "熊装",
    "睡帽",
    "雪地",
}
GOOD_NAME_HINTS = {
    "hero",
    "wizard",
    "magic",
    "ninja",
    "swordsman",
    "archer",
    "mecha",
    "pilot",
    "android",
    "warrior",
    "chef",
    "prince",
    "detective",
    "skate",
    "violin",
    "勇者",
    "冒险",
    "魔法",
    "忍者",
    "剑",
    "武士",
    "弓箭",
    "战士",
    "机械",
    "厨师",
    "王子",
    "侦探",
    "滑板",
    "小提琴",
    "歌手",
}


def iter_images() -> list[Path]:
    paths: list[Path] = []
    for root in SOURCE_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            low = path.stem.lower()
            if any(k.lower() in low for k in BAD_NAME_HINTS):
                continue
            paths.append(path)
    return sorted(set(paths))


def load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    dark = arr < 128
    light = arr > 127
    border_dark = np.r_[dark[0], dark[-1], dark[:, 0], dark[:, -1]].mean()
    mask = light if border_dark > 0.5 else dark
    mask = binary_closing(mask, iterations=1)
    mask = binary_fill_holes(mask)
    lab, n = label(mask)
    if n > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        mask = lab == int(counts.argmax())
    return np.asarray(mask, dtype=bool)


def contour_turn_score(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.012 * peri, True)
    return float(len(approx))


def metrics(path: Path) -> dict:
    mask = load_mask(path)
    area_px = int(mask.sum())
    if area_px == 0:
        return {"path": str(path), "valid": False, "reason": "empty"}
    lab, components = label(mask)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest_ratio = float(counts.max() / max(area_px, 1)) if len(counts) else 0.0
    ys, xs = np.where(mask)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    bbox_fill = float(area_px / max(w * h, 1))
    area = float(area_px / (SIZE * SIZE))
    aspect = float(h / max(w, 1))
    dist = distance_transform_edt(mask)
    thin_ratio = float((mask & (dist < 2.0)).sum() / area_px)
    robust_core = float((dist >= 3.0).sum() / area_px)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    complexity = float(perimeter / max(math.sqrt(area_px), 1.0))
    holes = 0 if hierarchy is None else int(sum(1 for hrow in hierarchy[0] if hrow[3] >= 0))
    opened = binary_opening(mask, iterations=2)
    ornament_loss = float(max(area_px - int(opened.sum()), 0) / area_px)
    top = mask[: int(SIZE * 0.38), :]
    mid = mask[int(SIZE * 0.30) : int(SIZE * 0.75), :]
    bottom = mask[int(SIZE * 0.74) :, :]
    top_ratio = float(top.sum() / area_px)
    mid_ratio = float(mid.sum() / area_px)
    bottom_ratio = float(bottom.sum() / area_px)
    left = int(mask[:, : SIZE // 2].sum())
    right = int(mask[:, SIZE // 2 :].sum())
    asymmetry = float(abs(left - right) / max(area_px, 1))
    turns = contour_turn_score(mask)
    name_bonus = 4.0 if any(k.lower() in path.stem.lower() for k in GOOD_NAME_HINTS) else 0.0

    penalties = 0.0
    penalties += max(0.0, 0.08 - area) * 120
    penalties += max(0.0, area - 0.34) * 70
    penalties += max(0.0, 0.18 - bbox_fill) * 35
    penalties += max(0.0, bbox_fill - 0.58) * 18
    penalties += max(0.0, thin_ratio - 0.43) * 40
    penalties += max(0.0, 0.12 - robust_core) * 45
    penalties += max(0.0, complexity - 23.0) * 1.1
    penalties += max(0.0, 7 - turns) * 1.5
    penalties += max(0.0, turns - 48) * 0.45
    penalties += max(0.0, holes - 5) * 1.8
    penalties += max(0.0, ornament_loss - 0.18) * 45
    penalties += max(0.0, 0.07 - top_ratio) * 40
    penalties += max(0.0, 0.16 - bottom_ratio) * 18
    penalties += max(0.0, 1.05 - aspect) * 7
    penalties += max(0.0, aspect - 2.65) * 8
    distinct_bonus = min(8.0, max(0.0, turns - 9) * 0.45) + min(5.0, asymmetry * 10.0)
    score = 100.0 - penalties + distinct_bonus + name_bonus
    pass_screen = (
        0.08 <= area <= 0.34
        and components <= 2
        and largest_ratio >= 0.98
        and 0.17 <= bbox_fill <= 0.62
        and thin_ratio <= 0.46
        and robust_core >= 0.11
        and complexity <= 26.0
        and holes <= 7
        and ornament_loss <= 0.23
        and 0.95 <= aspect <= 2.85
        and turns >= 8
    )
    return {
        "path": str(path),
        "name": path.stem,
        "valid": True,
        "pass_screen": bool(pass_screen),
        "score": round(float(score), 4),
        "area": round(area, 6),
        "bbox_fill": round(bbox_fill, 6),
        "aspect_h_over_w": round(aspect, 6),
        "components": int(components),
        "largest_ratio": round(largest_ratio, 6),
        "thin_ratio": round(thin_ratio, 6),
        "robust_core_ratio": round(robust_core, 6),
        "perimeter_complexity": round(complexity, 6),
        "holes": holes,
        "ornament_loss_open2": round(ornament_loss, 6),
        "top_ratio": round(top_ratio, 6),
        "mid_ratio": round(mid_ratio, 6),
        "bottom_ratio": round(bottom_ratio, 6),
        "asymmetry": round(asymmetry, 6),
        "contour_turns": round(turns, 3),
    }


def save_mask(path: Path, dst: Path) -> None:
    mask = load_mask(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).resize((512, 512), Image.Resampling.NEAREST).save(dst)


def make_sheet(rows: list[dict], out: Path, n: int = 80) -> None:
    rows = rows[:n]
    cell = 120
    label_h = 46
    cols = 8
    canvas = Image.new("RGB", (cols * cell, ((len(rows) + cols - 1) // cols) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x = (i % cols) * cell
        y = (i // cols) * (cell + label_h)
        mask = load_mask(Path(row["path"]))
        img = Image.fromarray(mask.astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(img, (x, y))
        draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(120, 120, 120))
        draw.text((x + 2, y + cell + 1), f"{i+1:02d} S={row['score']:.1f} T={row['contour_turns']:.0f}", fill=(0, 0, 0))
        draw.text((x + 2, y + cell + 17), row["name"][:18], fill=(0, 0, 0))
    canvas.save(out)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "selected").mkdir(parents=True)
    rows = [metrics(path) for path in iter_images()]
    valid = [row for row in rows if row.get("valid")]
    passed = [row for row in valid if row["pass_screen"]]
    passed.sort(key=lambda r: (-r["score"], r["thin_ratio"], -r["contour_turns"]))
    selected: list[dict] = []
    seen_names: set[str] = set()
    for row in passed:
        coarse = row["name"].split("_", 1)[-1].replace("的", "").replace("剪影", "").lower()
        if coarse in seen_names:
            continue
        selected.append(row)
        seen_names.add(coarse)
        if len(selected) >= 80:
            break
    for i, row in enumerate(selected, 1):
        save_mask(Path(row["path"]), OUT / "selected" / f"{i:03d}_{Path(row['path']).stem}.png")
    make_sheet(selected, OUT / "selected_contact_sheet.png")
    make_sheet(passed, OUT / "passed_contact_sheet.png")
    result = {
        "schema": "distinct_anime_silhouette_screen_20260709",
        "source_dirs": [str(p) for p in SOURCE_DIRS],
        "total_images": len(rows),
        "valid_images": len(valid),
        "passed_count": len(passed),
        "selected_count": len(selected),
        "selected_dir": str(OUT / "selected"),
        "selected_contact_sheet": str(OUT / "selected_contact_sheet.png"),
        "passed_contact_sheet": str(OUT / "passed_contact_sheet.png"),
        "selected": selected,
        "passed_top120": passed[:120],
    }
    (OUT / "screen_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "total_images": len(rows),
        "valid_images": len(valid),
        "passed_count": len(passed),
        "selected_count": len(selected),
        "selected_dir": str(OUT / "selected"),
        "selected_contact_sheet": str(OUT / "selected_contact_sheet.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
