"""Convert anime silhouette sources to black-background white silhouettes.

The simplification is intentionally mild: an elliptic open/close filter
removes hairline fragments and tiny notches while retaining the pose and
primary outline.  Original assets are never overwritten.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SOURCE = Path("/home/lin/xvxinyao/shadow/anime_shadow")
OUT = Path("/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified")
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    border = np.concatenate((gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()))
    background_is_light = float(np.median(border)) >= 127.0
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (threshold == 0 if background_is_light else threshold == 255).astype(np.uint8) * 255


def simplify(mask: np.ndarray) -> np.ndarray:
    side = min(mask.shape)
    kernel_size = 5 if side >= 700 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = max(24, int(cleaned.size * 0.00002))
    out = np.zeros_like(cleaned)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            out[labels == component] = 255
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    for index, source in enumerate(files, 1):
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        raw = foreground_mask(gray)
        result = simplify(raw)
        target = OUT / f"A{index:03d}_{source.stem}.png"
        cv2.imwrite(str(target), result)
        rows.append((f"A{index:03d}", target.name, source.name, int(np.count_nonzero(raw)), int(np.count_nonzero(result))))

    with (OUT / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("id", "processed", "source", "foreground_before_px", "foreground_after_px"))
        writer.writerows(rows)

    cell_w, cell_h, cols = 156, 184, 6
    pages = OUT / "contact_sheets"
    pages.mkdir(exist_ok=True)
    for offset in range(0, len(rows), cols * 6):
        subset = rows[offset:offset + cols * 6]
        sheet = Image.new("RGB", (cols * cell_w, ((len(subset) + cols - 1) // cols) * cell_h), "black")
        draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(subset):
            im = Image.open(OUT / row[1]).convert("L")
            im.thumbnail((142, 142))
            x, y = (local % cols) * cell_w, (local // cols) * cell_h
            sheet.paste(im.convert("RGB"), (x + 7, y + 5))
            draw.text((x + 5, y + 151), row[0], fill="white")
        sheet.save(pages / f"page_{offset // (cols * 6) + 1:03d}.jpg", quality=95)
    print(f"processed={len(rows)} out={OUT}")


if __name__ == "__main__":
    main()
