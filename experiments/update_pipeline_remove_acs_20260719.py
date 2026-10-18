#!/usr/bin/env python3
"""Replace the obsolete ACS panel in the raster pipeline with PINO+FEM ranking."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_aaai_submission/fig2_shadowforge_pipeline_final_imagegen_20260714.png"
OUTPUT = ROOT / "paper_aaai_submission/figures/fig2_shadowforge_pipeline_pino_fem_20260719.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{family}", size)


def centered(draw: ImageDraw.ImageDraw, box, text, size, fill, bold=False):
    f = font(size, bold)
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=f, spacing=2, align="center")
    x = (left + right - (bounds[2] - bounds[0])) / 2
    y = (top + bottom - (bounds[3] - bounds[1])) / 2
    draw.multiline_text((x, y), text, font=f, fill=fill, spacing=2, align="center")


def arrow(draw, start, end, color, width=3):
    draw.line([start, end], fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 > x0 else -1
        points = [(x1, y1), (x1 - sign * 11, y1 - 6), (x1 - sign * 11, y1 + 6)]
    else:
        sign = 1 if y1 > y0 else -1
        points = [(x1, y1), (x1 - 6, y1 - sign * 11), (x1 + 6, y1 - sign * 11)]
    draw.polygon(points, fill=color)


def main():
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Covers only the obsolete structural-screening panel and its former exits.
    panel = (1184, 108, 1630, 410)
    draw.rounded_rectangle(panel, radius=14, fill="white", outline="#90A4AE", width=2)
    centered(draw, (1202, 120, 1612, 152), "PINO-Guided Structural Screening", 15, "#263238", True)
    centered(draw, (1202, 152, 1612, 175), "optional self-weight ranking with FEM verification", 10, "#455A64")

    teal = "#00897B"
    amber = "#E58A00"
    gray = "#455A64"
    pale_teal = "#E8F5F3"
    pale_amber = "#FFF3D5"
    pale_gray = "#F2F4F5"
    boxes = [(1206, 191, 1303, 250), (1331, 191, 1428, 250), (1456, 191, 1554, 250)]
    colors = [(pale_gray, gray), (pale_teal, teal), (pale_amber, amber)]
    labels = ["candidate\nproxy", "PINO\nranks", "Top-k voxel\nFEM"]
    for b, (fill, outline), label in zip(boxes, colors, labels):
        draw.rounded_rectangle(b, radius=8, fill=fill, outline=outline, width=2)
        centered(draw, b, label, 12, "#263238", True)
    arrow(draw, (1304, 220), (1330, 220), gray, 2)
    arrow(draw, (1429, 220), (1455, 220), gray, 2)

    shortlist = (1284, 280, 1480, 324)
    draw.rounded_rectangle(shortlist, radius=8, fill="#EAF6F3", outline=teal, width=2)
    centered(draw, shortlist, "FEM-verified shortlist", 12, teal, True)
    centered(draw, (1210, 336, 1608, 367), "PINO prioritizes; voxel FEM remains final authority", 11, "#455A64")

    # Existing gate input remains aligned at the left panel edge. Make its intended entry explicit.
    arrow(draw, (1162, 188), (1206, 188), teal, 3)
    # Verified candidates inform the existing fidelity-band selector below.
    arrow(draw, (1382, 325), (1281, 430), teal, 3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
