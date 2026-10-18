#!/usr/bin/env python3
"""Render the four Figure 4 meshes in one matched three-receiver scene."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = "adventurer_schoolgirl_fighter_original"
BASE = ROOT / "results/fig4_anime_curated_original_v1_20260724"
SAD_ROOT = Path("/data/xvxinyao/ShadowArt-Revisited")
OUT = (
    ROOT
    / "paper_aaai_submission/fig4_single_column_bw_roi_20260726"
    / "blender_corner_renders"
)

METHODS = {
    "mitra": BASE / "mitra" / CASE / "mitra_pauly2009_shadow_hull.obj",
    "sadekar_mesh": (
        SAD_ROOT
        / "mesh_results/fig4_anime_curated_original_v1_sadekar_20260725"
        / "mesh"
        / CASE
        / "seed_17/sample_0_output.obj"
    ),
    "sadekar_voxel": (
        SAD_ROOT
        / "voxel_results/fig4_anime_curated_original_v1_sadekar_20260725"
        / "voxel"
        / CASE
        / "seed_17/sample_0_output.obj"
    ),
    "pcraft": (
        ROOT
        / "results/fig4_anime_pcraft_final_fixed3_pino_20260725_run2"
        / "adventurer_schoolgirl_fighter"
        / "selected_display_fixed.stl"
    ),
}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.82,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def import_mesh(path: Path) -> bpy.types.Object:
    if path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix.lower() == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        raise ValueError(path)
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "Reconstruction"
    return obj


def normalize_mesh(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    coords = [vertex.co.copy() for vertex in obj.data.vertices]
    low = Vector(
        (
            min(point.x for point in coords),
            min(point.y for point in coords),
            min(point.z for point in coords),
        )
    )
    high = Vector(
        (
            max(point.x for point in coords),
            max(point.y for point in coords),
            max(point.z for point in coords),
        )
    )
    center = (low + high) * 0.5
    scale = 2.15 / max(high.x - low.x, high.y - low.y, high.z - low.z, 1e-8)
    for vertex in obj.data.vertices:
        vertex.co.x = (vertex.co.x - center.x) * scale
        vertex.co.y = (vertex.co.y - center.y) * scale
        vertex.co.z = (vertex.co.z - low.z) * scale + 0.03
    bpy.context.view_layer.update()


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_spot(
    name: str,
    location: tuple[float, float, float],
    target: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
) -> None:
    bpy.ops.object.light_add(type="SPOT", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color = color
    light.data.spot_size = 0.56
    light.data.spot_blend = 0.34
    light.data.shadow_soft_size = 0.045
    point_at(light, Vector(target))


def setup_scene(obj: bpy.types.Object) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 680
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.film_transparent = False

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.985,
        0.985,
        0.985,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18
    scene.world = world

    floor_mat = make_material("Floor", (0.77, 0.82, 0.85, 1.0))
    left_mat = make_material("LeftWall", (0.98, 0.95, 0.62, 1.0))
    right_mat = make_material("RightWall", (0.85, 0.73, 0.91, 1.0))
    object_mat = make_material("Object", (0.66, 0.12, 0.035, 1.0), 0.62)
    obj.data.materials.clear()
    obj.data.materials.append(object_mat)

    add_box("Floor", (0.0, 0.0, -0.055), (2.75, 2.75, 0.055), floor_mat)
    add_box("LeftWall", (-2.75, 0.0, 2.70), (0.055, 2.75, 2.75), left_mat)
    add_box("RightWall", (0.0, 2.75, 2.70), (2.75, 0.055, 2.75), right_mat)

    # Each spotlight illuminates one receiver and casts a real mesh shadow.
    add_spot(
        "LeftReceiverLight",
        (5.0, 0.0, 1.18),
        (-2.75, 0.0, 1.18),
        1900.0,
        (1.0, 0.96, 0.82),
    )
    add_spot(
        "RightReceiverLight",
        (0.0, -5.0, 1.18),
        (0.0, 2.75, 1.18),
        1900.0,
        (0.94, 0.91, 1.0),
    )
    add_spot(
        "FloorReceiverLight",
        (0.0, 0.0, 6.0),
        (0.0, 0.0, 0.0),
        1550.0,
        (0.90, 0.96, 1.0),
    )

    bpy.ops.object.light_add(type="AREA", location=(3.0, -3.5, 5.0))
    fill = bpy.context.object
    fill.name = "ObjectFill"
    fill.data.energy = 190.0
    fill.data.shape = "DISK"
    fill.data.size = 3.0
    point_at(fill, Vector((0.0, 0.0, 0.85)))

    bpy.ops.object.camera_add(location=(6.5, -7.5, 5.3))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 7.25
    point_at(camera, Vector((-0.15, 0.15, 1.75)))
    scene.camera = camera


def render_method(slug: str, path: Path) -> None:
    clear_scene()
    obj = import_mesh(path)
    normalize_mesh(obj)
    setup_scene(obj)
    bpy.context.scene.render.filepath = str(OUT / f"{slug}_corner.png")
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {slug}: {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in METHODS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    selected = set(sys.argv[sys.argv.index("--") + 1 :]) if "--" in sys.argv else set()
    for slug, path in METHODS.items():
        if selected and slug not in selected:
            continue
        render_method(slug, path)
    print(OUT)


if __name__ == "__main__":
    main()
