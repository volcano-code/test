#!/usr/bin/env python3
"""Render a non-paper preview from raw local controlled-baseline projections.

This utility deliberately does not edit the paper.  It visualizes the exact
saved target/projection PNGs and the corresponding evaluator rows, so protocol
issues remain visible instead of being hidden by manually selected artwork.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "physical_print_controlled_baselines_v2_20260715"
OUT = ROOT / "results" / "fig3_raw_local_preview_20260716"
CASES = ("dark_knight_magic_scholar", "sword_fox")

METHODS = (
    ("Target shadows", None, None),
    ("Mitra-style hull", "mitra_style_carving", None),
    ("Sadekar mesh (seed 17)", "sadekar_mesh_based", "17"),
    ("Sadekar voxel (seed 17)", "sadekar_voxel_based", "17"),
    ("ShadowForge", "shadowforge", None),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def open_binary(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("L")
    # Preserve the stored silhouette rather than thresholding a second time.
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return Image.merge("RGB", (image, image, image))


def method_key(label: str) -> str:
    return {
        "Mitra-style hull": "Mitra-style carving",
        "Sadekar mesh (seed 17)": "Sadekar mesh-based",
        "Sadekar voxel (seed 17)": "Sadekar voxel-based",
        "ShadowForge": "ShadowForge",
    }[label]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "benchmark_manifest.json").read_text())
    cases = {row["case_id"]: row for row in manifest["cases"]}

    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with (SOURCE / "per_run_results.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[(row["case_id"], row["method"], row["seed"])] = row

    tile = 154
    gap = 12
    label_w = 270
    panel_w = 2 * tile + 3 * gap
    top_h = 104
    row_h = tile + 46
    footer_h = 64
    width = label_w + len(CASES) * panel_w + (len(CASES) + 1) * gap
    height = top_h + len(METHODS) * row_h + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for j, case_id in enumerate(CASES):
        x0 = label_w + gap + j * panel_w
        title = case_id.replace("_", " / ")
        draw.text((x0 + panel_w // 2, 26), title, anchor="ma", font=font(20, True), fill="#17212b")
        draw.text((x0 + tile // 2, 66), "view X", anchor="ma", font=font(14), fill="#455a64")
        draw.text((x0 + 2 * gap + 3 * tile // 2, 66), "view Y", anchor="ma", font=font(14), fill="#455a64")

    for i, (label, directory, seed) in enumerate(METHODS):
        y0 = top_h + i * row_h
        draw.line((label_w, y0, width - gap, y0), fill="#d5dce0", width=1)
        draw.text((label_w - 18, y0 + tile // 2), label, anchor="rm", font=font(16, True), fill="#17212b")
        for j, case_id in enumerate(CASES):
            x0 = label_w + gap + j * panel_w
            if directory is None:
                paths = [Path(p) for p in cases[case_id]["input_paths"]]
            else:
                base = SOURCE / "evaluated_projections" / case_id / directory
                prefix = f"seed_{seed}_" if seed else "deterministic_"
                paths = [base / f"{prefix}view0.png", base / f"{prefix}view1.png"]
            for k, path in enumerate(paths):
                image = open_binary(path, tile)
                px = x0 + k * (tile + gap)
                canvas.paste(image, (px, y0 + 4))
                draw.rectangle((px, y0 + 4, px + tile - 1, y0 + tile + 3), outline="#aeb8bd", width=1)
            if directory is not None:
                row = rows[(case_id, method_key(label), seed or "")]
                draw.text(
                    (x0 + panel_w // 2, y0 + tile + 20),
                    f"raw mIoU {float(row['mean_iou']):.3f}   Dice {float(row['mean_dice']):.3f}",
                    anchor="ma", font=font(13), fill="#455a64",
                )

    footer_y = height - footer_h + 18
    draw.line((gap, footer_y - 12, width - gap, footer_y - 12), fill="#d5dce0", width=1)
    draw.text(
        (width // 2, footer_y),
        "Raw local projection preview only: metrics and volume QA have not yet passed the final shared-protocol audit.",
        anchor="ma", font=font(13, True), fill="#9a5b00",
    )
    draw.text(
        (width // 2, footer_y + 24),
        "Source: physical_print_controlled_baselines_v2_20260715; no paper assets were edited.",
        anchor="ma", font=font(12), fill="#455a64",
    )

    destination = OUT / "fig3_raw_local_two_case_preview.png"
    canvas.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
