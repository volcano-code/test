#!/usr/bin/env python3
"""Render audited target/projection galleries for the view-count matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/viewcount_matrix_anime_ordinary_mixed_20260730"
SUMMARY = OUTPUT / "SUMMARY.json"

CANVAS_WIDTH = 3200
LEFT_MARGIN = 42
RIGHT_MARGIN = 42
LABEL_WIDTH = 500
TOP_HEIGHT = 172
ROW_HEIGHT = 390
GAP = 18
TILE_BORDER = "#B0BEC5"
TEXT = "#263238"
MUTED = "#607D8B"
FAMILY_COLORS = {
    "anime": "#00897B",
    "ordinary": "#4F83CC",
    "mixed": "#E08A00",
}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def sorted_images(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        digits = "".join(character for character in path.stem if character.isdigit())
        return (int(digits) if digits else -1, path.name)

    return sorted(paths, key=key)


def image_pairs(result_dir: Path, views: int) -> list[tuple[Path, Path]]:
    candidates = [
        (
            sorted_images(list((result_dir / "inputs").glob("*target*.png"))),
            sorted_images(list((result_dir / "projections").glob("*projection*.png"))),
        ),
        (
            sorted_images(list(result_dir.glob("target_view_*.png"))),
            sorted_images(list(result_dir.glob("projection_view_*.png"))),
        ),
        (
            sorted_images(list((result_dir / "projections").glob("target_view_*.png"))),
            sorted_images(
                list((result_dir / "projections").glob("projection_view_*.png"))
            ),
        ),
        (
            sorted_images(list((result_dir / "soft_training").glob("gt_view_*.png"))),
            sorted_images(
                list((result_dir / "soft_training").glob("pred_view_*.png"))
            ),
        ),
    ]
    for targets, projections in candidates:
        if len(targets) >= views and len(projections) >= views:
            return list(zip(targets[:views], projections[:views]))
    return []


def percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def paste_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    path: Path,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill="white", outline=TILE_BORDER, width=2)
    with Image.open(path) as source:
        image = ImageOps.contain(
            source.convert("RGB"),
            (right - left - 12, bottom - top - 12),
            Image.Resampling.LANCZOS,
        )
    x = left + (right - left - image.width) // 2
    y = top + (bottom - top - image.height) // 2
    canvas.paste(image, (x, y))


def render(rows: list[dict[str, Any]], views: int, output_path: Path) -> None:
    valid_rows: list[tuple[dict[str, Any], list[tuple[Path, Path]]]] = []
    for row in rows:
        pairs = image_pairs(Path(row["result_dir"]), views)
        if pairs:
            valid_rows.append((row, pairs))

    height = TOP_HEIGHT + max(1, len(valid_rows)) * ROW_HEIGHT + 34
    canvas = Image.new("RGB", (CANVAS_WIDTH, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (LEFT_MARGIN, 26),
        f"P-CRAFT Audited {views}-View Examples",
        font=font(54, bold=True),
        fill=TEXT,
    )
    draw.text(
        (LEFT_MARGIN, 96),
        "Each pair shows the prescribed silhouette and delivered-model projection.",
        font=font(27),
        fill=MUTED,
    )

    grid_left = LEFT_MARGIN + LABEL_WIDTH
    usable_width = CANVAS_WIDTH - grid_left - RIGHT_MARGIN
    tile_gap_count = views * 2 - 1
    tile_width = (usable_width - tile_gap_count * GAP) // (views * 2)
    for view in range(views):
        pair_left = grid_left + view * (tile_width * 2 + GAP * 2)
        draw.text(
            (pair_left + tile_width - 72, 123),
            f"View {view + 1}",
            font=font(23, bold=True),
            fill=TEXT,
        )

    for row_index, (row, pairs) in enumerate(valid_rows):
        top = TOP_HEIGHT + row_index * ROW_HEIGHT
        color = FAMILY_COLORS.get(row["family"], MUTED)
        draw.line(
            (LEFT_MARGIN, top, CANVAS_WIDTH - RIGHT_MARGIN, top),
            fill="#CFD8DC",
            width=2,
        )
        draw.rounded_rectangle(
            (LEFT_MARGIN, top + 28, LEFT_MARGIN + 16, top + ROW_HEIGHT - 34),
            radius=8,
            fill=color,
        )
        draw.text(
            (LEFT_MARGIN + 38, top + 46),
            str(row["family"]).upper(),
            font=font(28, bold=True),
            fill=color,
        )
        label = str(row["id"]).replace(f"{row['family']}_{views}v_", "")
        draw.multiline_text(
            (LEFT_MARGIN + 38, top + 92),
            label.replace("_", " "),
            font=font(29, bold=True),
            fill=TEXT,
            spacing=8,
        )
        draw.multiline_text(
            (LEFT_MARGIN + 38, top + 202),
            (
                f"mean IoU  {percent(row.get('mean_iou'))}\n"
                f"min IoU    {percent(row.get('min_iou'))}\n"
                f"Dice       {percent(row.get('mean_dice'))}\n"
                f"saving     {percent(row.get('saving'))}"
            ),
            font=font(23),
            fill=MUTED,
            spacing=8,
        )

        tile_top = top + 38
        tile_bottom = top + ROW_HEIGHT - 34
        for view, (target, projection) in enumerate(pairs):
            target_left = grid_left + view * (tile_width * 2 + GAP * 2)
            prediction_left = target_left + tile_width + GAP
            paste_tile(
                canvas,
                draw,
                target,
                (target_left, tile_top, target_left + tile_width, tile_bottom),
            )
            paste_tile(
                canvas,
                draw,
                projection,
                (
                    prediction_left,
                    tile_top,
                    prediction_left + tile_width,
                    tile_bottom,
                ),
            )
            draw.text(
                (target_left + 10, tile_top + 8),
                "Target",
                font=font(19, bold=True),
                fill="#455A64",
                stroke_width=3,
                stroke_fill="white",
            )
            draw.text(
                (prediction_left + 10, tile_top + 8),
                "Projection",
                font=font(19, bold=True),
                fill="#455A64",
                stroke_width=3,
                stroke_fill="white",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(300, 300), optimize=True)


def main() -> None:
    if not SUMMARY.is_file():
        raise SystemExit(f"Missing summary: {SUMMARY}")
    summary = json.loads(SUMMARY.read_text())
    rows = [
        row
        for row in summary["matrix"]
        if row.get("status") != "PENDING" and row.get("mean_iou") is not None
    ]
    render(
        [row for row in rows if int(row["views"]) == 2],
        2,
        OUTPUT / "audited_two_view_examples.png",
    )
    render(
        [row for row in rows if int(row["views"]) == 3],
        3,
        OUTPUT / "audited_three_view_examples.png",
    )
    print(OUTPUT / "audited_two_view_examples.png")
    print(OUTPUT / "audited_three_view_examples.png")


if __name__ == "__main__":
    main()
