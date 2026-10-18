#!/usr/bin/env python3
"""Add the deterministic von-Mises and safety-factor tail to Figure 3."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_aaai_submission/figures/pino_architecture_raster_simplified_2x_20260718.png"
OUTPUT = ROOT / "paper_aaai_submission/figures/pino_architecture_raster_with_vm_sf_20260719.png"


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSerif.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def arrow(draw, a, b, fill="#263238", width=8):
    draw.line((a, b), fill=fill, width=width)
    x, y = b
    draw.polygon([(x, y), (x - 24, y - 13), (x - 24, y + 13)], fill=fill)


def center(draw, xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def main():
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # The original figure reserves a wide blank right margin after the two field outputs.
    # Add only the deterministic tail, retaining every existing network module.
    x = int(w * 0.795)
    y_stress = int(h * 0.485)
    y_vm = int(h * 0.605)
    y_sf = int(h * 0.725)
    charcoal = "#263238"
    violet = "#5E35B1"

    # Clear the unused margin where the deterministic chain is placed.
    draw.rounded_rectangle((x - 85, y_stress - 45, w - 65, y_sf + 80), radius=18, fill="white")
    # Reconnect from the existing physical-stress output on the left of the blank margin.
    arrow(draw, (int(w * 0.708), y_stress), (x - 28, y_stress), charcoal, 7)

    center(draw, (x + 22, y_stress), "σ̂", font(70, True), violet)
    arrow(draw, (x + 22, y_stress + 48), (x + 22, y_vm - 38), charcoal, 7)
    center(draw, (x + 22, y_vm), "σ̂_vm", font(58, True), violet)
    arrow(draw, (x + 22, y_vm + 45), (x + 22, y_sf - 42), charcoal, 7)
    center(draw, (x + 72, int((y_vm + y_sf) / 2)), "max", font(32), charcoal)
    center(draw, (x + 22, y_sf), "ŜF", font(70, True), violet)

    # Small explanatory labels are kept intentionally brief to preserve column readability.
    sans = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 31)
    draw.text((x + 95, y_vm - 15), "von Mises", font=sans, fill=charcoal)
    draw.text((x + 95, y_sf - 15), "derived safety factor", font=sans, fill=charcoal)

    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
