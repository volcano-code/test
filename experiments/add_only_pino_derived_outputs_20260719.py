#!/usr/bin/env python3
"""Append only the deterministic stress-to-safety output tail to Figure 3."""
from pathlib import Path

from io import BytesIO

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_aaai_submission/figures/pino_architecture_raster_simplified_2x_20260718.png"
OUTPUT = ROOT / "paper_aaai_submission/figures/pino_architecture_raster_simplified_with_derived_outputs_20260719.png"


def sans(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def text_center(draw, center, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center[0] - (box[2] - box[0]) / 2, center[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)


def down_arrow(draw, x, start_y, end_y, color, width=7):
    draw.line((x, start_y, x, end_y), fill=color, width=width)
    draw.polygon([(x, end_y + 15), (x - 12, end_y - 10), (x + 12, end_y - 10)], fill=color)


def math_label(expression: str) -> Image.Image:
    """Render with one LaTeX-style math family matching the existing outputs."""
    stream = BytesIO()
    math_to_image(
        f"${expression}$",
        stream,
        prop=FontProperties(size=28),
        dpi=160,
        format="png",
        color="#111111",
    )
    stream.seek(0)
    return Image.open(stream).convert("RGBA")


def paste_center(image: Image.Image, label: Image.Image, center):
    image.alpha_composite(label, (round(center[0] - label.width / 2), round(center[1] - label.height / 2)))


def main():
    image = Image.open(SOURCE).convert("RGBA")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    charcoal = "#263238"
    # Existing u-hat and sigma-hat are centered at x ~= 2818 in the source raster.
    # Continue from exactly that centerline; no existing pixel is altered.
    x = 2818
    sigma_vm_y = 1310
    sf_y = 1590
    vm = math_label(r"\hat{\sigma}_{\mathrm{vm}}")
    sf = math_label(r"\widehat{SF}")

    down_arrow(draw, x, 1092, sigma_vm_y - vm.height // 2 - 18, charcoal, width=6)
    paste_center(image, vm, (x, sigma_vm_y))
    down_arrow(draw, x, sigma_vm_y + vm.height // 2 + 18, sf_y - sf.height // 2 - 18, charcoal, width=6)
    paste_center(image, sf, (x, sf_y))

    image.convert("RGB").save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
