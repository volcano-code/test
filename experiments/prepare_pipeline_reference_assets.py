#!/usr/bin/env python3
"""Collect exact PNG evidence for the ShadowForge pipeline figure."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps
from skimage.measure import marching_cubes

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
E = ROOT / "results/dark_knight_fdm_7cm_evidence_20260713"
OUT = ROOT / "paper_aaai_submission/pipeline_reference_assets_20260714"
BLENDER = Path("/home/lin/.local/bin/blender")


def copy(src, name):
    dst = OUT / name
    shutil.copy2(src, dst)
    print(dst)


def black_background_to_alpha(src, dst):
    image = Image.open(src).convert("RGBA")
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, _ = pixels[x, y]
            if r < 24 and g < 24 and b < 24:
                pixels[x, y] = (r, g, b, 0)
    image.save(dst)
    print(dst)


def npy_to_ply(src, dst, threshold=0.5):
    volume = np.squeeze(np.load(src)).astype(np.float32)
    if not (volume.min() < threshold < volume.max()):
        threshold = float(np.percentile(volume, 90))
    vertices, faces, _, _ = marching_cubes(np.pad(volume, 1), level=threshold)
    vertices -= vertices.mean(axis=0, keepdims=True)
    vertices /= max(np.ptp(vertices, axis=0).max(), 1e-6)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(dst)


def make_projection_compare(target_x, target_y, pred_x, pred_y, output):
    """Make a contact sheet in the same image convention as the pipeline inputs."""
    panels = [
        ("Target X", target_x), ("Target Y", target_y),
        ("Rendered X", pred_x), ("Rendered Y", pred_y),
    ]
    tile, label_h = 256, 28
    sheet = Image.new("RGB", (tile * 2, (tile + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.NEAREST)
        x, y = (index % 2) * tile, (index // 2) * (tile + label_h)
        sheet.paste(image, (x + (tile - image.width) // 2, y + label_h))
        draw.text((x + 8, y + 7), label, fill="black")
    sheet.save(output)
    print(output)


def flip_render_vertical(path):
    """Match Blender's X-view screen-up convention to the wall-image Z axis."""
    image = Image.open(path).convert("RGBA")
    ImageOps.flip(image).save(path)
    print(path)


def render_mesh(mesh_path, output, material, emission=False, camera=None):
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
bsdf.inputs['Metallic'].default_value = 0.0
if __EMISSION__:
    bsdf.inputs['Emission Color'].default_value = (__R__, __G__, __B__, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 0.18
obj.data.materials.append(mat)
cam_data = bpy.data.cameras.new('Camera')
cam = bpy.data.objects.new('Camera', cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
scale = max(obj.dimensions) if max(obj.dimensions) > 0 else 1.0
cam.location = tuple(c * scale for c in (__CAMERA__))
cam.rotation_euler = (Vector((0,0,0)) - cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type = 'ORTHO'; cam.data.ortho_scale = 1.65 * scale
for name, loc, energy in [('Key',(1.5,-2,3.5),700), ('Fill',(-2,1,2),350)]:
    ld = bpy.data.lights.new(name, type='AREA'); ld.energy = energy; ld.shape='DISK'; ld.size=4
    lo = bpy.data.objects.new(name, ld); bpy.context.collection.objects.link(lo); lo.location = loc
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 700; scene.render.resolution_y = 700; scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True
scene.render.filepath = r'__OUTPUT__'
bpy.ops.render.render(write_still=True)
""".replace("__INPUT__", str(mesh_path)).replace("__OUTPUT__", str(output))
    script = (script.replace("__R__", str(material[0]))
                   .replace("__G__", str(material[1]))
                   .replace("__B__", str(material[2]))
                   .replace("__EMISSION__", str(emission))
                   .replace("__CAMERA__", repr(camera or (2.0, -2.4, 1.65))))
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "render.py"
        script_path.write_text(script)
        subprocess.run([str(BLENDER), "--background", "--python", str(script_path)], check=True)
    print(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scaffold = E / "candidate_evidence/delivery_candidates/soft_guided_full_hull_scaffold_core48_delivery256"
    carved = E / "candidate_evidence/delivery_candidates/soft_guided_carve_phase_1_removed_2544_core48_delivery256"
    for src, name in [
        (E / "inputs/left_wall_x_target.png", "01_target_x.png"),
        (E / "inputs/right_wall_y_target.png", "02_target_y.png"),
        (E / "soft_training/pred_view_0.png", "03_soft_render_x.png"),
        (E / "soft_training/pred_view_1.png", "04_soft_render_y.png"),
        (scaffold / "projection_view1.png", "07_scaffold_projection_x.png"),
        (scaffold / "projection_view2.png", "07_scaffold_projection_y.png"),
        (scaffold / "target_vs_projection.png", "07_scaffold_compare.png"),
        (carved / "projection_view1.png", "09_carved_projection_x.png"),
        (carved / "projection_view2.png", "09_carved_projection_y.png"),
        (carved / "target_vs_projection.png", "09_carved_compare.png"),
        (E / "input_vs_projection.png", "11_final_comparison.png"),
    ]:
        copy(src, name)
    # Do not reuse the legacy wall-axis contact sheets: their panel layout is
    # intentionally different from the normalized image convention exported above.
    make_projection_compare(
        OUT / "01_target_x.png", OUT / "02_target_y.png",
        OUT / "07_scaffold_projection_x.png", OUT / "07_scaffold_projection_y.png",
        OUT / "07_scaffold_compare.png",
    )
    make_projection_compare(
        OUT / "01_target_x.png", OUT / "02_target_y.png",
        OUT / "09_carved_projection_x.png", OUT / "09_carved_projection_y.png",
        OUT / "09_carved_compare.png",
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        soft_ply, scaffold_ply, carved_ply = tmp / "soft.ply", tmp / "scaffold.ply", tmp / "carved.ply"
        npy_to_ply(E / "soft_probability_256.npy", soft_ply, 0.05)
        npy_to_ply(scaffold / "volume.npy", scaffold_ply)
        npy_to_ply(carved / "volume.npy", carved_ply)
        # Use the exact X projection for the soft field. The near-X final mesh
        # render is intentionally more three-dimensional, but this proposal
        # panel must be directly comparable with the X target silhouette.
        render_mesh(soft_ply, OUT / "05_soft_occupancy.png", (0.02, 0.42, 0.38),
                    camera=(2.5, 0.0, 0.0))
        flip_render_vertical(OUT / "05_soft_occupancy.png")
        render_mesh(scaffold_ply, OUT / "06_scaffold_3d.png", (0.12, 0.48, 0.52))
        render_mesh(carved_ply, OUT / "08_carved_3d.png", (0.88, 0.48, 0.10))
    # Render the selected delivery occupancy, rather than its raw STL triangles:
    # the latter repeats vertices per face and gives Blender unreliable local normals.
    with tempfile.TemporaryDirectory() as tmp:
        final_ply = Path(tmp) / "selected_delivery.ply"
        npy_to_ply(E / "selected_display_voxels256.npy", final_ply)
        # A near-X projection preserves the left-wall silhouette while retaining
        # enough yaw/elevation to show the object is a 3D fabrication candidate.
        x_view = (2.4, -0.65, 0.45)
        y_view = (0.65, -2.4, 0.45)
        render_mesh(final_ply, OUT / "10_final_model.png", (0.20, 0.42, 0.68), camera=x_view)
        render_mesh(final_ply, OUT / "10_final_model_view_x.png", (0.20, 0.42, 0.68), camera=x_view)
        render_mesh(final_ply, OUT / "10_final_model_view_y.png", (0.20, 0.42, 0.68), camera=y_view)
        render_mesh(final_ply, OUT / "10_final_model_projection_x.png", (0.20, 0.42, 0.68), camera=(2.5, 0.0, 0.0))


if __name__ == "__main__":
    main()
