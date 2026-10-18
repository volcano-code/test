#!/usr/bin/env python3
"""Audit and assemble the third fixed-XYZ anime three-view gallery."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results" / "fixed_xyz_carve_anime_gallery_3_20260731"
)
CASES = (
    ("Campus motion", "campus_motion"),
    ("Urban allies", "urban_allies"),
    ("Future stealth", "future_stealth"),
)
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def audit_mesh(path: Path) -> dict[str, object]:
    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    parts = mesh.split(only_watertight=False)
    return {
        "audit_watertight": bool(mesh.is_watertight),
        "audit_winding_consistent": bool(mesh.is_winding_consistent),
        "audit_mesh_parts": int(len(parts)),
        "audit_vertices": int(len(mesh.vertices)),
        "audit_faces": int(len(mesh.faces)),
        "audit_volume_mm3": float(abs(mesh.volume)),
        "audit_extents_mm": [float(value) for value in mesh.extents],
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    records: list[dict[str, object]] = []
    card_width = 1040
    card_height = 1010
    gap = 28
    margin = 36
    canvas = Image.new(
        "RGB",
        (3 * card_width + 2 * gap + 2 * margin, card_height + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(str(FONT_BOLD), 31)
    metric_font = ImageFont.truetype(str(FONT_REGULAR), 21)
    small_font = ImageFont.truetype(str(FONT_REGULAR), 18)

    for index, (display_name, directory_name) in enumerate(CASES):
        case_dir = output_root / directory_name
        manifest = json.loads((case_dir / "MANIFEST.json").read_text())
        selected = manifest["selected"]
        stl_path = case_dir / "selected_display.stl"
        audit = audit_mesh(stl_path)
        record = {
            "case": directory_name,
            "status": manifest["status"],
            "candidate": selected["candidate"],
            "mean_iou": selected["mean_iou"],
            "worst_view_iou": selected["min_iou"],
            "mean_dice": selected["mean_dice"],
            "saving_percent": 100.0 * selected["material_saved_vs_hull"],
            "single_body": selected["component_count_6"] == 1,
            "manifest_watertight": selected["mesh_watertight"],
            "erosion_survival": selected["erosion_survival"],
            "runtime_seconds": manifest["runtime_seconds"],
            "stl": str(stl_path),
            **audit,
        }
        records.append(record)

        x0 = margin + index * (card_width + gap)
        y0 = margin
        audit_ok = (
            record["single_body"]
            and record["audit_watertight"]
            and record["audit_winding_consistent"]
            and record["audit_mesh_parts"] == 1
        )
        outline = "#00897B" if manifest["status"] == "PASS" and audit_ok else "#EF6C00"
        draw.rounded_rectangle(
            (x0, y0, x0 + card_width, y0 + card_height),
            radius=8,
            fill="#FFFFFF",
            outline=outline,
            width=3,
        )
        draw.text((x0 + 22, y0 + 18), display_name, fill="#263238", font=title_font)
        metrics = (
            f"{manifest['status']}  IoU {selected['mean_iou']:.4f}  "
            f"Worst {selected['min_iou']:.4f}  "
            f"Saving {100.0 * selected['material_saved_vs_hull']:.2f}%"
        )
        draw.text((x0 + 22, y0 + 62), metrics, fill="#455A64", font=metric_font)
        topology = (
            f"Independent STL audit: one body {'yes' if record['single_body'] else 'no'}, "
            f"watertight {'yes' if audit_ok else 'no'}"
        )
        draw.text((x0 + 22, y0 + 94), topology, fill="#455A64", font=small_font)

        render_path = case_dir / "selected_render.png"
        if render_path.is_file():
            render = fit_image(Image.open(render_path).convert("RGB"), 660, 560)
            render_x = x0 + (card_width - render.width) // 2
            canvas.paste(render, (render_x, y0 + 130))

        audit_image = fit_image(
            Image.open(case_dir / "selected_projection_audit.png").convert("RGB"),
            970,
            300,
        )
        audit_x = x0 + (card_width - audit_image.width) // 2
        canvas.paste(audit_image, (audit_x, y0 + 690))

    (output_root / "SUMMARY.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    with (output_root / "SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    canvas.save(output_root / "anime_gallery_3_evidence.png")

    readme_lines = [
        "# P-CRAFT anime three-view gallery 3",
        "",
        "All cases use the isolated fixed_xyz_carve branch at 256^3.",
        "The branch performs fixed-axis visual-hull construction, target-ray repair,",
        "connectivity-preserving carving, printable thickening, and SDF meshing.",
        "PINO/FEM structural screening is not run in this visual-geometry gallery.",
        "",
    ]
    for record in records:
        readme_lines.extend([
            f"## {record['case']}",
            f"- Status: {record['status']}",
            f"- Mean / worst-view IoU: {record['mean_iou']:.4f} / {record['worst_view_iou']:.4f}",
            f"- Material saving vs. visual hull: {record['saving_percent']:.2f}%",
            f"- Independent STL audit: parts={record['audit_mesh_parts']}, "
            f"watertight={record['audit_watertight']}, "
            f"winding_consistent={record['audit_winding_consistent']}",
            f"- STL: `{record['stl']}`",
            "",
        ])
    (output_root / "README.md").write_text("\n".join(readme_lines))


if __name__ == "__main__":
    main()
