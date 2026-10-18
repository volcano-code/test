#!/usr/bin/env python3
"""Build a labeled contact sheet for the 24 urban-allies room orientations."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "urban_allies/orientation_candidates"
)
OUTPUT = SOURCE / "contact_sheet.png"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def main() -> None:
    manifest = json.loads((SOURCE / "orientations.json").read_text(encoding="utf-8"))
    records = manifest["orientations"]
    columns = 4
    rows = (len(records) + columns - 1) // columns
    tile_width = 640
    tile_height = 323
    label_height = 78
    gap = 14
    margin = 24

    width = margin * 2 + columns * tile_width + (columns - 1) * gap
    height = margin * 2 + rows * (tile_height + label_height) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(25)
    matrix_font = load_font(21)

    for position, record in enumerate(records):
        row, column = divmod(position, columns)
        x = margin + column * (tile_width + gap)
        y = margin + row * (tile_height + label_height + gap)
        render = Image.open(record["render"]).convert("RGB")
        if render.size != (tile_width, tile_height):
            render = render.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(render, (x, y))
        draw.rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            outline=(80, 80, 80),
            width=2,
        )
        basis = record["basis"]
        compact_basis = "; ".join(
            " ".join(f"{value:+d}" for value in row) for row in basis
        )
        draw.text(
            (x + 8, y + tile_height + 7),
            f"Orientation {record['orientation']:02d}",
            fill=(32, 39, 43),
            font=title_font,
        )
        draw.text(
            (x + 8, y + tile_height + 39),
            compact_basis,
            fill=(69, 82, 88),
            font=matrix_font,
        )

    sheet.save(OUTPUT, dpi=(220, 220))
    print(OUTPUT)


if __name__ == "__main__":
    main()
