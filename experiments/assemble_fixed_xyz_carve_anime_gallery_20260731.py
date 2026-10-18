#!/usr/bin/env python3
"""Assemble deterministic evidence for the fixed-XYZ carved-cube branch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "fixed_xyz_carve_anime_gallery_20260731"
CASES = (
    ("Arcane casters", "arcane_casters"),
    ("Blade heroes", "blade_heroes"),
    ("Stage dancers", "stage_dancers"),
    ("Beast spirits", "beast_spirits"),
)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def main() -> None:
    records: list[dict[str, object]] = []
    card_width = 1460
    card_height = 680
    gap = 32
    margin = 48
    canvas = Image.new(
        "RGB",
        (2 * card_width + gap + 2 * margin, 2 * card_height + gap + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(FONT_BOLD), 31)
    metric_font = ImageFont.truetype(str(FONT_REGULAR), 23)

    for index, (display_name, directory_name) in enumerate(CASES):
        case_dir = OUTPUT_ROOT / directory_name
        manifest = json.loads((case_dir / "MANIFEST.json").read_text())
        selected = manifest["selected"]
        record = {
            "case": directory_name,
            "candidate": selected["candidate"],
            "mean_iou": selected["mean_iou"],
            "worst_view_iou": selected["min_iou"],
            "saving_percent": 100.0 * selected["material_saved_vs_hull"],
            "single_body": selected["component_count_6"] == 1,
            "watertight": selected["mesh_watertight"],
            "erosion_survival": selected["erosion_survival"],
            "runtime_seconds": manifest["runtime_seconds"],
            "stl": str(case_dir / "selected_display.stl"),
            "projection_audit": str(case_dir / "selected_projection_audit.png"),
            "render": str(case_dir / "selected_render.png"),
        }
        records.append(record)

        column = index % 2
        row = index // 2
        x0 = margin + column * (card_width + gap)
        y0 = margin + row * (card_height + gap)
        draw.rounded_rectangle(
            (x0, y0, x0 + card_width, y0 + card_height),
            radius=8,
            fill="#FFFFFF",
            outline="#B0BEC5",
            width=2,
        )
        draw.text((x0 + 24, y0 + 18), display_name, fill="#263238", font=title_font)
        metrics = (
            f"Mean IoU {selected['mean_iou']:.4f}   "
            f"Worst view {selected['min_iou']:.4f}   "
            f"Saving {100.0 * selected['material_saved_vs_hull']:.2f}%   "
            "Single / watertight"
        )
        draw.text((x0 + 24, y0 + 60), metrics, fill="#455A64", font=metric_font)

        render = fit_image(Image.open(case_dir / "selected_render.png").convert("RGB"), 570, 560)
        audit = fit_image(
            Image.open(case_dir / "selected_projection_audit.png").convert("RGB"),
            800,
            560,
        )
        render_x = x0 + 24 + (570 - render.width) // 2
        audit_x = x0 + 624 + (800 - audit.width) // 2
        image_y = y0 + 105
        canvas.paste(render, (render_x, image_y + (560 - render.height) // 2))
        canvas.paste(audit, (audit_x, image_y + (560 - audit.height) // 2))

    (OUTPUT_ROOT / "SUMMARY.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    with (OUTPUT_ROOT / "SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    canvas.save(OUTPUT_ROOT / "anime_gallery_evidence.png")


if __name__ == "__main__":
    main()
