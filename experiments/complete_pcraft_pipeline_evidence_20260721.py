#!/usr/bin/env python3
"""Build traceable P-CRAFT pipeline assets from existing experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OLD_ASSETS = ROOT / "paper_aaai_submission/pipeline_reference_assets_current_20260718"
RUN = ROOT / "results/dark_knight_fdm_7cm_evidence_20260713"
CANDIDATES = RUN / "candidate_evidence/delivery_candidates"
STRUCTURAL = ROOT / "results/pino_ranked_fem_taskdisjoint_v16_20260720"
OUT = ROOT / "paper_aaai_submission/p_craft_pipeline_evidence_20260721"

TRAJECTORY = [
    (
        "06_repaired_scaffold_3d.png",
        "soft_guided_full_hull_scaffold_core48_delivery256",
        (0.16, 0.55, 0.55),
        "Repaired scaffold",
    ),
    (
        "07_carve_heavy_3d.png",
        "soft_guided_carve_phase_1_removed_2544_core48_delivery256",
        (0.89, 0.60, 0.22),
        "Carve phase 1",
    ),
    (
        "08_carve_middle_3d.png",
        "soft_guided_carve_phase_2_removed_1045_core48_delivery256",
        (0.91, 0.48, 0.12),
        "Carve phase 2",
    ),
    (
        "09_carve_light_3d.png",
        "soft_guided_carve_phase_3_removed_37_core48_delivery256",
        (0.78, 0.34, 0.08),
        "Carve phase 3",
    ),
]


def load_render_helpers():
    source = ROOT / "experiments/prepare_current_pipeline_assets_20260718.py"
    spec = importlib.util.spec_from_file_location("pcraft_render_helpers", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load render helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.volume_to_mesh, module.render_mesh


def copy_asset(name: str) -> Path:
    source = OLD_ASSETS / name
    destination = OUT / name
    shutil.copy2(source, destination)
    return source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(family, size)


def centered(draw: ImageDraw.ImageDraw, box, text: str, text_font, fill) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
    )


def rounded_box(draw, box, outline, fill, radius=18, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill, width=5):
    draw.line((start, end), fill=fill, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 14, y - 8), (x - 14, y + 8)], fill=fill)


def make_structural_panel(summary_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text())
    policy = summary["policy"]
    sealed = summary["sealed_test"]
    image = Image.new("RGB", (1800, 700), "white")
    draw = ImageDraw.Draw(image)
    teal = "#00897B"
    amber = "#FFB300"
    gray = "#455A64"
    pale = "#ECEFF1"
    dark = "#263238"

    draw.text((54, 36), "PINO-guided Top-k voxel FEM", font=font(46, True), fill=dark)
    draw.text(
        (54, 94),
        "Task-disjoint sealed-test evidence; no ACS decision stage",
        font=font(27),
        fill=gray,
    )

    boxes = [
        ((70, 190, 390, 360), "Fidelity-band\ncandidates", gray, pale),
        ((520, 190, 840, 360), "PINO\nranks", teal, "#E0F2F1"),
        ((970, 190, 1290, 360), f"Top-{policy['top_k']}\nvoxel FEM", amber, "#FFF3CD"),
        ((1420, 190, 1740, 360), "Verified\nselection", teal, "#E3F4EA"),
    ]
    for box, label, outline, fill in boxes:
        rounded_box(draw, box, outline, fill)
        for offset, line in enumerate(label.splitlines()):
            line_box = (box[0], box[1] + 38 + offset * 52, box[2], box[1] + 90 + offset * 52)
            centered(draw, line_box, line, font(31, True), dark)
    for first, second in zip(boxes, boxes[1:]):
        arrow(draw, (first[0][2] + 16, 275), (second[0][0] - 18, 275), dark)

    values = [
        ("Sealed tasks", str(policy["sealed_tasks"])),
        ("FEM calls", f"{sealed['fem_calls']} / {sealed['full_fem_calls']}"),
        ("FEM reduction", f"{100 * sealed['fem_call_reduction_rate']:.1f}%"),
        ("Exact match", f"{sealed['exact_lightest_safe_matches']} / {policy['sealed_tasks']}"),
    ]
    x_positions = [70, 505, 940, 1375]
    for x, (label, value) in zip(x_positions, values):
        rounded_box(draw, (x, 455, x + 355, 625), "#CFD8DC", "white", radius=14, width=2)
        centered(draw, (x, 475, x + 355, 520), label, font(25), gray)
        centered(draw, (x, 525, x + 355, 600), value, font(38, True), teal)
    image.save(output)


def make_trajectory_sheet(rendered: list[Path], output: Path) -> None:
    canvas = Image.new("RGBA", (1800, 520), "white")
    draw = ImageDraw.Draw(canvas)
    labels = [item[3] for item in TRAJECTORY]
    for index, (path, label) in enumerate(zip(rendered, labels)):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((380, 380), Image.Resampling.LANCZOS)
        x = 30 + index * 445
        canvas.alpha_composite(image, (x + (380 - image.width) // 2, 18))
        centered(draw, (x, 410, x + 380, 472), label, font(27, True), "#263238")
        if index < len(rendered) - 1:
            arrow(draw, (x + 392, 215), (x + 435, 215), "#FFB300", width=5)
    canvas.convert("RGB").save(output)


def extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        result = volume[index, :, :]
    elif axis == 1:
        result = volume[:, index, :]
    else:
        result = volume[:, :, index]
    return np.flipud(result.T)


def make_material_reduction_panel(rendered: list[Path], output: Path) -> None:
    volumes = []
    stages = []
    for _, candidate_name, _, _ in TRAJECTORY:
        volumes.append(np.squeeze(np.load(CANDIDATES / candidate_name / "volume.npy")) > 0.5)
        stages.append(json.loads((CANDIDATES / candidate_name / "stage.json").read_text()))

    removed = volumes[0] & ~volumes[-1]
    slice_scores = []
    for axis in range(3):
        summed = removed.sum(axis=tuple(i for i in range(3) if i != axis))
        margin = max(2, len(summed) // 12)
        index = int(np.argmax(summed[margin:-margin]) + margin)
        slice_scores.append((int(summed[index]), axis, index))
    _, axis, index = max(slice_scores)

    canvas = Image.new("RGB", (1800, 940), "white")
    draw = ImageDraw.Draw(canvas)
    labels = [item[3] for item in TRAJECTORY]
    base_count = int(volumes[0].sum())
    for panel, (path, label, volume, stage) in enumerate(zip(rendered, labels, volumes, stages)):
        x = 18 + panel * 445
        model = Image.open(path).convert("RGBA")
        model.thumbnail((350, 350), Image.Resampling.LANCZOS)
        canvas.paste(model, (x + (390 - model.width) // 2, 10), model)
        centered(draw, (x, 355, x + 390, 402), label, font(25, True), "#263238")

        current_slice = extract_slice(volume, axis, index)
        removed_slice = extract_slice(volumes[0] & ~volume, axis, index)
        rgb = np.full((*current_slice.shape, 3), 255, dtype=np.uint8)
        rgb[current_slice] = (38, 103, 112) if panel == 0 else (185, 112, 32)
        rgb[removed_slice] = (210, 45, 67)
        section = Image.fromarray(rgb, mode="RGB").resize((330, 330), Image.Resampling.NEAREST)
        canvas.paste(section, (x + 30, 430))
        draw.rectangle((x + 30, 430, x + 360, 760), outline="#90A4AE", width=2)

        count = int(volume.sum())
        removed_pct = 100.0 * (base_count - count) / max(base_count, 1)
        centered(draw, (x, 775, x + 390, 817), f"Occupied: {count:,}", font(22), "#455A64")
        centered(
            draw,
            (x, 815, x + 390, 855),
            f"Removed vs scaffold: {removed_pct:.1f}%",
            font(22, True),
            "#D22D43" if panel else "#455A64",
        )
        centered(
            draw,
            (x, 855, x + 390, 897),
            f"Reported saving: {100 * stage['saving']:.1f}%",
            font(22, True),
            "#00897B",
        )

        if panel < len(rendered) - 1:
            arrow(draw, (x + 392, 188), (x + 435, 188), "#FFB300", width=5)

    draw.text((28, 905), "Cross-section: retained material", font=font(20), fill="#455A64")
    draw.rectangle((310, 909, 336, 931), fill="#B97020")
    draw.text((346, 905), "removed material", font=font(20), fill="#455A64")
    draw.rectangle((530, 909, 556, 931), fill="#D22D43")
    draw.text(
        (590, 905),
        f"Shared slice: axis {'XYZ'[axis]}, index {index}",
        font=font(20),
        fill="#607D8B",
    )
    canvas.save(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}

    for name in [
        "01_target_x.png",
        "02_target_y.png",
        "03_soft_render_x.png",
        "04_soft_render_y.png",
        "05_soft_occupancy.png",
        "09_selected_compare.png",
        "10_final_model.png",
    ]:
        sources[name] = str(copy_asset(name))

    volume_to_mesh, render_mesh = load_render_helpers()
    rendered = []
    camera = (2.4, -0.65, 0.45)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for output_name, candidate_name, color, _ in TRAJECTORY:
            source = CANDIDATES / candidate_name / "volume.npy"
            mesh = directory / f"{candidate_name}.ply"
            destination = OUT / output_name
            volume_to_mesh(source, mesh, 0.5)
            render_mesh(mesh, destination, color, camera)
            rendered.append(destination)
            sources[output_name] = str(source)

    make_trajectory_sheet(rendered, OUT / "10_guarded_carving_trajectory.png")
    sources["10_guarded_carving_trajectory.png"] = "composite of 06-09 from one saved trajectory"
    make_material_reduction_panel(rendered, OUT / "10_guarded_carving_material_evidence.png")
    sources["10_guarded_carving_material_evidence.png"] = (
        "same-camera renders and shared cross-section from the 06-09 saved volumes"
    )

    structural_summary = STRUCTURAL / "summary.json"
    make_structural_panel(structural_summary, OUT / "11_pino_topk_fem_evidence.png")
    sources["11_pino_topk_fem_evidence.png"] = str(structural_summary)

    for _, candidate_name, _, _ in TRAJECTORY:
        stage = CANDIDATES / candidate_name / "stage.json"
        destination = OUT / f"metrics_{candidate_name}.json"
        shutil.copy2(stage, destination)
        sources[destination.name] = str(stage)
    shutil.copy2(structural_summary, OUT / "metrics_pino_topk_fem.json")
    sources["metrics_pino_topk_fem.json"] = str(structural_summary)

    records = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.json":
            records.append(
                {
                    "asset": path.name,
                    "source": sources.get(path.name, "generated from listed traceable assets"),
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema": "pcraft_pipeline_evidence_v1",
        "method_name": "P-CRAFT",
        "notes": [
            "All carving renders use one saved dark-knight trajectory and one camera.",
            "The structural panel reports raw PINO ranking plus Top-k voxel FEM; ACS is absent.",
            "No metrics were synthesized or manually altered.",
        ],
        "assets": records,
    }
    (OUT / "EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    readme = """# P-CRAFT pipeline evidence

This directory contains traceable assets for the compact P-CRAFT overview figure.

- `01`-`05`: targets and differentiable soft evidence.
- `06`-`09`: one real repaired-to-carved trajectory rendered with an identical camera.
- `10_guarded_carving_trajectory.png`: trajectory contact sheet.
- `10_guarded_carving_material_evidence.png`: same-position cross-sections and real material counts.
- `11_pino_topk_fem_evidence.png`: no-ACS PINO ranking and Top-k FEM evidence.
- `09_selected_compare.png` and `10_final_model.png`: final projection and geometry evidence.

See `EVIDENCE_MANIFEST.json` for source paths and checksums.
"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps({"output": str(OUT), "assets": len(records)}, indent=2))


if __name__ == "__main__":
    main()
