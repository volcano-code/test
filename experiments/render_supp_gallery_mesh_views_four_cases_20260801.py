#!/usr/bin/env python3
"""Render protocol-axis mesh evidence for gallery rows three and four."""

from __future__ import annotations

from pathlib import Path
import os

import bpy
from mathutils import Matrix, Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
GALLERY = ROOT / "paper_aaai_submission/supplement_gallery_one_case_20260801"
STRUCT_CASE = Path(
    "/data/xvxinyao/xvxinyao/StructShadow/print_ready_stls_diversity/by_group/"
    "n3_g5_ant_dragon_shrimp"
)
CASES = (
    {
        "name": "diverse5_thick1_13_21_23",
        "mesh": ROOT
        / "results/anime30_diverse_fixed_50_60_10cm_256_20260707"
        / "diverse5_thick1_13_21_23"
        / "diverse5_thick1_13_21_23_carve_pf4_10cm_256.stl",
    },
    {
        "name": "ant_dragon_shrimp",
        "mesh": STRUCT_CASE / "g5_bugs_ant_x_dragon_sea_shrimp_MINIMAL.stl",
    },
    {
        "name": "arcane_casters",
        "mesh": ROOT
        / "results/fixed_xyz_carve_anime_gallery_20260731"
        / "arcane_casters"
        / "selected_display.stl",
    },
    {
        "name": "urban_allies",
        "mesh": ROOT
        / "results/fixed_xyz_carve_anime_gallery_3_final_20260731"
        / "urban_allies"
        / "selected_display.stl",
    },
)
TEAL = (0.035, 0.31, 0.33, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_mesh(path: Path) -> bpy.types.Object:
    bpy.ops.wm.stl_import(filepath=str(path))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def normalize_mesh(obj: bpy.types.Object, target_size: float = 1.58) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    points = [vertex.co.copy() for vertex in obj.data.vertices]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (low + high) * 0.5
    scale = target_size / max(high - low)
    for vertex in obj.data.vertices:
        vertex.co.x = (vertex.co.x - center.x) * scale
        vertex.co.y = (vertex.co.y - center.y) * scale
        vertex.co.z = (vertex.co.z - center.z) * scale


def material() -> bpy.types.Material:
    result = bpy.data.materials.new("UnifiedDeepTeal")
    result.use_nodes = True
    result.diffuse_color = TEAL
    bsdf = result.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = TEAL
    bsdf.inputs["Roughness"].default_value = 0.68
    return result


def set_camera_basis(camera, right, up, back) -> None:
    camera.rotation_euler = Matrix(
        (
            (right[0], up[0], back[0]),
            (right[1], up[1], back[1]),
            (right[2], up[2], back[2]),
        )
    ).to_euler()


def add_light(location, energy, size) -> None:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    light.rotation_euler = (Vector((0.0, 0.0, 0.0)) - light.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()


VIEWS = (
    ((-4.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    ((0.0, -4.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.0, 0.0, 4.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)


def render_view(case: dict, index: int) -> None:
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    scene.world = world

    obj = import_mesh(case["mesh"])
    normalize_mesh(obj)
    obj.data.materials.clear()
    obj.data.materials.append(material())
    add_light((3.8, -4.2, 5.0), 1100.0, 4.0)
    add_light((-3.0, 2.4, 2.8), 500.0, 3.0)

    location, right, up, back = VIEWS[index]
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.90
    set_camera_basis(camera, right, up, back)
    scene.camera = camera
    out = GALLERY / case["name"] / "renders_four_case_centered"
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / f"mesh_view_{index}.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    selected = os.environ.get("PCRAFT_GALLERY_CASE")
    for case in CASES:
        if selected and case["name"] != selected:
            continue
        if not case["mesh"].is_file():
            raise FileNotFoundError(case["mesh"])
        for index in range(3):
            render_view(case, index)
        print(case["name"])


if __name__ == "__main__":
    main()
