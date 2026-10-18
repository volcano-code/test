#!/usr/bin/env python3
"""Prepare traceable current-method evidence for the compact pipeline figure."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.measure import marching_cubes


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
RUN = ROOT / "results/dark_knight_current_softseed_evidence_20260718"
SOFT = ROOT / "results/dark_knight_fdm_7cm_fdmrank_20260713/soft_training"
OUT = ROOT / "paper_aaai_submission/pipeline_reference_assets_current_20260718"
BLENDER = Path("/home/lin/.local/bin/blender")

RAW = RUN / "candidate_evidence/seed_t0_55_raw"
RESTORED = RUN / "candidate_evidence/seed_t0_55_restore_rays"
REPAIRED = RUN / "candidate_evidence/seed_t0_55_bridge6"
SELECTED = (
    RUN
    / "candidate_evidence/delivery_candidates"
    / "soft_seed_t0_55_carve_phase_3_removed_11_core48_delivery256"
)


def copy_file(source: Path, name: str) -> Path:
    destination = OUT / name
    shutil.copy2(source, destination)
    return destination


def volume_to_mesh(source: Path, destination: Path, threshold: float = 0.5) -> None:
    volume = np.squeeze(np.load(source)).astype(np.float32)
    if not (float(volume.min()) < threshold < float(volume.max())):
        threshold = float(np.percentile(volume, 90))
    vertices, faces, _, _ = marching_cubes(np.pad(volume, 1), level=threshold)
    vertices -= vertices.mean(axis=0, keepdims=True)
    vertices /= max(float(np.ptp(vertices, axis=0).max()), 1e-6)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(destination)


def render_mesh(
    mesh_path: Path,
    output: Path,
    color: tuple[float, float, float],
    camera: tuple[float, float, float] = (2.4, -0.65, 0.45),
) -> None:
    script = """import bpy
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
path = r'__INPUT__'
if path.lower().endswith('.stl'):
    bpy.ops.wm.stl_import(filepath=path)
else:
    bpy.ops.wm.ply_import(filepath=path)
obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)
mat = bpy.data.materials.new('EvidenceMaterial')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (__R__, __G__, __B__, 1.0)
bsdf.inputs['Roughness'].default_value = 0.58
obj.data.materials.append(mat)
cam_data = bpy.data.cameras.new('Camera')
cam = bpy.data.objects.new('Camera', cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
scale = max(obj.dimensions) if max(obj.dimensions) > 0 else 1.0
cam.location = tuple(c * scale for c in __CAMERA__)
cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 1.58 * scale
for name, loc, energy in [('Key', (1.5, -2, 3.5), 700), ('Fill', (-2, 1, 2), 350)]:
    data = bpy.data.lights.new(name, type='AREA')
    data.energy = energy
    data.shape = 'DISK'
    data.size = 4
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = loc
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 700
scene.render.resolution_y = 700
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True
scene.render.filepath = r'__OUTPUT__'
bpy.ops.render.render(write_still=True)
"""
    script = (
        script.replace("__INPUT__", str(mesh_path))
        .replace("__OUTPUT__", str(output))
        .replace("__R__", str(color[0]))
        .replace("__G__", str(color[1]))
        .replace("__B__", str(color[2]))
        .replace("__CAMERA__", repr(camera))
    )
    with tempfile.TemporaryDirectory() as directory:
        script_path = Path(directory) / "render.py"
        script_path.write_text(script)
        subprocess.run(
            [str(BLENDER), "--background", "--python", str(script_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def make_comparison(target_x: Path, target_y: Path, pred_x: Path, pred_y: Path, output: Path) -> None:
    panels = [
        ("Target X", target_x),
        ("Target Y", target_y),
        ("Rendered X", pred_x),
        ("Rendered Y", pred_y),
    ]
    tile, label_height = 256, 30
    sheet = Image.new("RGB", (tile * 2, (tile + label_height) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.NEAREST)
        x = (index % 2) * tile
        y = (index // 2) * (tile + label_height)
        sheet.paste(image, (x + (tile - image.width) // 2, y + label_height))
        draw.text((x + 8, y + 8), label, fill="black")
    sheet.save(output)


def make_contact_sheet(paths: list[Path], labels: list[str], output: Path) -> None:
    tile, label_height = 300, 42
    columns = 4
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGBA", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (path, label) in enumerate(zip(paths, labels)):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((tile - 20, tile - 20), Image.Resampling.LANCZOS)
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        sheet.alpha_composite(image, (x + (tile - image.width) // 2, y + 10))
        draw.text((x + 10, y + tile + 8), label, fill="#263238", font=font)
    sheet.convert("RGB").save(output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    target_x = copy_file(RUN / "inputs/left_wall_x_target.png", "01_target_x.png")
    target_y = copy_file(RUN / "inputs/right_wall_y_target.png", "02_target_y.png")
    copy_file(SOFT / "pred_view_0.png", "03_soft_render_x.png")
    copy_file(SOFT / "pred_view_1.png", "04_soft_render_y.png")

    copy_file(REPAIRED / "projection_view1.png", "07_repaired_projection_x.png")
    copy_file(REPAIRED / "projection_view2.png", "07_repaired_projection_y.png")
    make_comparison(
        target_x,
        target_y,
        OUT / "07_repaired_projection_x.png",
        OUT / "07_repaired_projection_y.png",
        OUT / "07_repaired_compare.png",
    )

    copy_file(SELECTED / "projection_view1.png", "09_selected_projection_x.png")
    copy_file(SELECTED / "projection_view2.png", "09_selected_projection_y.png")
    make_comparison(
        target_x,
        target_y,
        OUT / "09_selected_projection_x.png",
        OUT / "09_selected_projection_y.png",
        OUT / "09_selected_compare.png",
    )
    copy_file(RUN / "input_vs_projection.png", "11_final_comparison.png")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        meshes = {
            "soft": (RUN / "soft_probability_256.npy", 0.05),
            "raw": (RAW / "volume.npy", 0.5),
            "restored": (RESTORED / "volume.npy", 0.5),
            "repaired": (REPAIRED / "volume.npy", 0.5),
            "selected": (SELECTED / "volume.npy", 0.5),
            "final": (RUN / "selected_display_voxels256.npy", 0.5),
        }
        ply = {}
        for name, (source, threshold) in meshes.items():
            ply[name] = directory / f"{name}.ply"
            volume_to_mesh(source, ply[name], threshold)

        render_mesh(ply["soft"], OUT / "05_soft_occupancy.png", (0.02, 0.42, 0.38), (2.5, 0.0, 0.0))
        ImageOps.flip(Image.open(OUT / "05_soft_occupancy.png").convert("RGBA")).save(
            OUT / "05_soft_occupancy.png"
        )
        render_mesh(ply["repaired"], OUT / "06_repaired_candidate_3d.png", (0.12, 0.48, 0.52))
        render_mesh(ply["selected"], OUT / "08_selected_delivery_3d.png", (0.88, 0.48, 0.10))
        render_mesh(ply["final"], OUT / "10_final_model.png", (0.20, 0.42, 0.68))
        render_mesh(ply["raw"], OUT / "12_raw_seed_3d.png", (0.45, 0.55, 0.60))
        render_mesh(ply["restored"], OUT / "13_ray_restored_3d.png", (0.25, 0.55, 0.48))

    copy_file(RAW / "target_vs_projection.png", "12_raw_seed_compare.png")
    copy_file(RESTORED / "target_vs_projection.png", "13_ray_restored_compare.png")
    copy_file(REPAIRED / "stage.json", "14_repaired_stage_metrics.json")
    copy_file(SELECTED / "stage.json", "15_selected_stage_metrics.json")

    primary = [
        OUT / "01_target_x.png",
        OUT / "02_target_y.png",
        OUT / "05_soft_occupancy.png",
        OUT / "06_repaired_candidate_3d.png",
        OUT / "07_repaired_compare.png",
        OUT / "08_selected_delivery_3d.png",
        OUT / "09_selected_compare.png",
        OUT / "10_final_model.png",
        OUT / "11_final_comparison.png",
    ]
    make_contact_sheet(
        primary,
        [
            "Target X",
            "Target Y",
            "Soft occupancy",
            "Repaired C48",
            "Repair projections",
            "Selected Vdel,256",
            "Selected projections",
            "Final model",
            "Final comparison",
        ],
        OUT / "00_contact_sheet.png",
    )

    manifest = json.loads((RUN / "MANIFEST.json").read_text())
    records = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "ASSET_MANIFEST.json":
            records.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    asset_manifest = {
        "pipeline_story": [
            "prescribed shadows",
            "differentiable soft proposal",
            "fixed-threshold soft seed with ray restoration and Bridge6",
            "guarded carving and 48-to-256 delivery",
            "fabrication QA and fidelity-band selection",
            "selected printable delivery",
        ],
        "run": str(RUN),
        "soft_artifact": str(SOFT),
        "selected_candidate": manifest["selected"]["candidate"],
        "selected_metrics": manifest["selected"],
        "structural_screen_note": (
            "Not requested in this evidence run; optional PINO/ACS/FEM must remain schematic "
            "unless a separately verified structural result is supplied."
        ),
        "files": records,
    }
    (OUT / "ASSET_MANIFEST.json").write_text(json.dumps(asset_manifest, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
