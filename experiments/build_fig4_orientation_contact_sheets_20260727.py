#!/usr/bin/env python3
"""Build labeled 8x6 contact sheets for Figure 4 orientation audits."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
AUDIT_ROOT = (
    ROOT
    / "paper_aaai_submission/fig4_table1_connected06_soft_preview_20260726"
    / "blender_true_shadow_renders/orientation_audit_all48"
)
METHODS = ("sadekar_mesh", "sadekar_voxel", "pcraft")
COLS, ROWS = 8, 6
THUMB_W, THUMB_H = 300, 225
LABEL_H = 28


def font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build(method: str) -> Path:
    source_dir = AUDIT_ROOT / method
    images = sorted(source_dir.glob("[0-9][0-9]_p*_s*.png"))
    if len(images) != 48:
        raise RuntimeError(f"{method}: expected 48 images, found {len(images)}")

    sheet = Image.new(
        "RGB",
        (COLS * THUMB_W, ROWS * (THUMB_H + LABEL_H)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    label_font = font(17)
    for index, path in enumerate(images):
        row, col = divmod(index, COLS)
        image = Image.open(path).convert("RGB")
        image.thumbnail((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
        x = col * THUMB_W + (THUMB_W - image.width) // 2
        y = row * (THUMB_H + LABEL_H)
        sheet.paste(image, (x, y))
        label = path.stem
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_x = col * THUMB_W + (THUMB_W - (bbox[2] - bbox[0])) // 2
        draw.text(
            (text_x, y + THUMB_H + 3),
            label,
            fill=(38, 50, 56),
            font=label_font,
        )
    output = source_dir / "contact_sheet.png"
    sheet.save(output, optimize=True)
    return output


def main() -> None:
    for method in METHODS:
        print(build(method))


if __name__ == "__main__":
    main()
