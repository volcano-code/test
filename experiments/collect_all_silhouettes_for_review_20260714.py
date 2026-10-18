"""Collect locally available silhouette assets for manual selection.

Copies only image files from known user/project locations, keeps a TSV mapping
to every original file, deduplicates exact byte-identical assets, and produces
numbered contact sheets for review.  Source files are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


OUT = Path("/home/lin/xvxinyao/all_silhouettes_for_manual_selection_20260714")
ROOTS = (
    Path("/home/lin/下载"),
    Path("/home/lin/xvxinyao/ShadowForge/results"),
    Path("/home/lin/xvxinyao/download_anime_character_silhouettes_20260707"),
    Path("/home/lin/xvxinyao/anime_silhouettes_all_20260707"),
    Path("/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2"),
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
EXCLUDE = {".cache", ".codex", ".local", ".config", "anaconda3", "node_modules", "__pycache__"}
TOKENS = ("silhouette", "shadow", "anime", "target", "input", "inputs", "gt_view", "sdlist")


def relevant(path: Path) -> bool:
    text = str(path).lower()
    return any(token in text for token in TOKENS)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_label(path: Path) -> str:
    words = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")
    return words[:56] or "silhouette"


def make_contact_sheets(rows: list[dict[str, str]], out: Path) -> None:
    thumb, cell_w, cell_h, cols = 112, 150, 144, 8
    per_page = cols * 8
    pages = out / "contact_sheets"
    pages.mkdir(exist_ok=True)
    for page_idx, offset in enumerate(range(0, len(rows), per_page), 1):
        page_rows = rows[offset:offset + per_page]
        n_rows = (len(page_rows) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, n_rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page_rows):
            x = (index % cols) * cell_w
            y = (index // cols) * cell_h
            try:
                im = Image.open(out / row["copy"]).convert("RGB")
                im.thumbnail((thumb, thumb))
                canvas = Image.new("RGB", (thumb, thumb), "white")
                canvas.paste(im, ((thumb - im.width) // 2, (thumb - im.height) // 2))
                sheet.paste(canvas, (x + 19, y + 4))
                draw.text((x + 4, y + 118), row["id"], fill="black")
                draw.text((x + 35, y + 118), row["label"][:16], fill="black")
            except Exception:
                draw.text((x + 4, y + 4), f"{row['id']} unreadable", fill="red")
        sheet.save(pages / f"page_{page_idx:03d}.jpg", quality=92)


def main() -> None:
    images = OUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    duplicates: list[tuple[str, str]] = []
    candidate_count = 0

    for root in ROOTS:
        if not root.exists():
            continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE]
            directory_path = Path(directory)
            for name in names:
                source = directory_path / name
                if source.suffix.lower() not in IMAGE_SUFFIXES or not relevant(source):
                    continue
                candidate_count += 1
                try:
                    key = digest(source)
                    if key in seen:
                        duplicates.append((str(source), seen[key]))
                        continue
                    image_id = f"S{len(rows) + 1:05d}"
                    copy_name = f"{image_id}_{safe_label(source)}{source.suffix.lower()}"
                    target = images / copy_name
                    shutil.copy2(source, target)
                    seen[key] = copy_name
                    rows.append({
                        "id": image_id,
                        "label": safe_label(source),
                        "copy": f"images/{copy_name}",
                        "source": str(source),
                        "sha256": key,
                    })
                except OSError:
                    continue

    with (OUT / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "label", "copy", "source", "sha256"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "duplicates.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("duplicate_source", "canonical_copy"))
        writer.writerows(duplicates)
    make_contact_sheets(rows, OUT)
    (OUT / "README.txt").write_text(
        f"Collected {len(rows)} unique silhouette-related images from {candidate_count} candidates.\n"
        "Use contact_sheets/page_*.jpg for review; manifest.tsv maps each ID to its original path.\n",
        encoding="utf-8",
    )
    print(f"unique={len(rows)} candidates={candidate_count} duplicates={len(duplicates)} out={OUT}")


if __name__ == "__main__":
    main()
