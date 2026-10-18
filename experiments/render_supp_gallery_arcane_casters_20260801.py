#!/usr/bin/env python3
"""Render evidence views for one P-CRAFT supplementary gallery example."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = ROOT / "results/fixed_xyz_carve_anime_gallery_20260731/arcane_casters"
MESH = CASE / "selected_display.stl"
OUT = ROOT / "paper_aaai_submission/supplement_gallery_one_case_20260801/renders"
SIMULATION_BASIS: Matrix | None = None
SIMULATION_ENABLE_FLOOR_PROJECTION = True
SIMULATION_BLEND_NAME = "simulation.blend"


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
    roughness: float = 0.72,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def import_mesh() -> bpy.types.Object:
    if not MESH.is_file():
        raise FileNotFoundError(MESH)
    bpy.ops.wm.stl_import(filepath=str(MESH))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {MESH}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "ArcaneCasters"
    return obj


def normalize_mesh(obj: bpy.types.Object, target_size: float = 1.55) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    points = [vertex.co.copy() for vertex in obj.data.vertices]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (low + high) * 0.5
    scale = target_size / max((high - low), key=abs)
    for vertex in obj.data.vertices:
        vertex.co.x = (vertex.co.x - center.x) * scale
        vertex.co.y = (vertex.co.y - center.y) * scale
        vertex.co.z = (vertex.co.z - low.z) * scale
    bpy.context.view_layer.update()


def point_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    obj.rotation_euler = (
        Vector(target) - obj.location
    ).to_track_quat("-Z", "Y").to_euler()


def set_camera_basis(
    camera: bpy.types.Object,
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    back: tuple[float, float, float],
) -> None:
    """Set an orthographic camera using the protocol's image-plane axes."""
    rotation = Matrix(
        (
            (right[0], up[0], back[0]),
            (right[1], up[1], back[1]),
            (right[2], up[2], back[2]),
        )
    )
    camera.rotation_euler = rotation.to_euler()


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def configure_world(transparent: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = transparent
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 0.32 if transparent else 0.18
    scene.world = world


def add_area_light(
    location: tuple[float, float, float],
    energy: float,
    size: float,
    target: tuple[float, float, float],
) -> None:
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    point_at(light, target)


def render_axis_view(
    index: int,
    camera_location: tuple[float, float, float],
    camera_right: tuple[float, float, float],
    camera_up: tuple[float, float, float],
    camera_back: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> None:
    clear_scene()
    configure_world(transparent=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100

    obj = import_mesh()
    normalize_mesh(obj)
    obj.data.materials.clear()
    obj.data.materials.append(make_material("Object", color, 0.66))

    add_area_light((3.8, -4.2, 5.0), 1050.0, 4.0, (0.0, 0.0, 0.72))
    add_area_light((-3.0, 2.4, 2.8), 450.0, 3.0, (0.0, 0.0, 0.72))

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.95
    set_camera_basis(camera, camera_right, camera_up, camera_back)
    scene.camera = camera
    scene.render.filepath = str(OUT / f"mesh_view_{index}.png")
    bpy.ops.render.render(write_still=True)


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
    light.data.spot_size = math.radians(36.0)
    light.data.spot_blend = 0.24
    light.data.shadow_soft_size = 0.025
    point_at(light, target)


def render_simulation() -> None:
    clear_scene()
    configure_world(transparent=False)
    scene = bpy.context.scene
    scene.render.resolution_x = 1120
    scene.render.resolution_y = 760
    scene.render.resolution_percentage = 100

    obj = import_mesh()
    normalize_mesh(obj, target_size=1.34)
    if SIMULATION_BASIS is not None:
        # Keep the evidence mesh unchanged, but express it in the physical
        # corner frame selected by the protocol.  This is a rigid axis change,
        # not a per-view image rotation.
        obj.rotation_euler = SIMULATION_BASIS.to_euler()
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    obj.location.x -= (low.x + high.x) * 0.5
    obj.location.y -= (low.y + high.y) * 0.5
    obj.location.z += 0.10 - low.z
    obj.data.materials.clear()
    obj.data.materials.append(
        make_material("PCRAFTAmber", (0.78, 0.31, 0.055, 1.0), 0.58)
    )

    floor_material = make_material("Floor", (0.76, 0.80, 0.82, 1.0), 0.86)
    left_material = make_material("LeftWall", (0.92, 0.90, 0.66, 1.0), 0.88)
    right_material = make_material("RightWall", (0.82, 0.76, 0.90, 1.0), 0.88)
    add_box("Floor", (0.0, 0.0, -0.06), (2.15, 2.15, 0.06), floor_material)
    add_box("LeftWall", (-2.15, 0.0, 2.10), (0.06, 2.15, 2.10), left_material)
    add_box("RightWall", (0.0, 2.15, 2.10), (2.15, 0.06, 2.10), right_material)

    add_spot(
        "ProjectionX",
        (4.4, 0.0, 1.12),
        (-2.15, 0.0, 1.12),
        1600.0,
        (1.0, 0.96, 0.84),
    )
    add_spot(
        "ProjectionY",
        (0.0, -4.4, 1.12),
        (0.0, 2.15, 1.12),
        1600.0,
        (0.94, 0.91, 1.0),
    )
    if SIMULATION_ENABLE_FLOOR_PROJECTION:
        add_spot(
            "ProjectionZ",
            (0.0, 0.0, 5.0),
            (0.0, 0.0, 0.0),
            1380.0,
            (0.92, 0.97, 1.0),
        )
    add_area_light((3.4, -3.5, 4.4), 220.0, 3.2, (0.0, 0.0, 0.72))

    bpy.ops.object.camera_add(location=(4.65, -5.35, 3.45))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 5.72
    point_at(camera, (-0.08, 0.10, 1.15))
    scene.camera = camera
    scene.render.filepath = str(OUT / "simulation.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / SIMULATION_BLEND_NAME))
    bpy.ops.render.render(write_still=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The fixed-TFS protocol stores x/y/z projections as [y,z], [x,z],
    # and [x,y]. These bases make Blender's screen axes match those arrays.
    render_axis_view(
        0,
        (4.0, 0.0, 0.78),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.20, 0.48, 0.72, 1.0),
    )
    render_axis_view(
        1,
        (0.0, -4.0, 0.78),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.13, 0.58, 0.52, 1.0),
    )
    render_axis_view(
        2,
        (0.0, 0.0, 4.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.45, 0.33, 0.69, 1.0),
    )
    render_simulation()
    print(OUT)


if __name__ == "__main__":
    main()
