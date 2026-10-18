#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


MITRA_PANEL = (46, 1768, 1292, 2577)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace only the bottom-left Mitra render in Figure 4."
    )
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure = Image.open(args.figure).convert("RGBA")
    render = Image.open(args.render).convert("RGBA")

    if figure.size != (2574, 3438):
        raise ValueError(f"Unexpected Figure 4 size: {figure.size}")

    x0, y0, x1, y1 = MITRA_PANEL
    panel_w = x1 - x0
    panel_h = y1 - y0
    scale = min(panel_w / render.width, panel_h / render.height)
    resized = render.resize(
        (round(render.width * scale), round(render.height * scale)),
        Image.Resampling.LANCZOS,
    )

    figure.paste((255, 255, 255, 255), MITRA_PANEL)
    paste_x = x0 + (panel_w - resized.width) // 2
    paste_y = y0 + (panel_h - resized.height) // 2
    figure.alpha_composite(resized, (paste_x, paste_y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.convert("RGB").save(args.output, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
