#!/usr/bin/env python3
"""Render an archived Sadekar OBJ through Blender's matching TFS views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--ortho-scale", type=float, default=2.0)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)


def import_obj(path: Path) -> bpy.types.Object:
    bpy.ops.wm.obj_import(filepath=str(path))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    result = bpy.context.view_layer.objects.active
    result.name = "OfficialSadekarGeometry"
    # Blender's OBJ importer adds a +90 degree X rotation for its default
    # up-axis conversion. Sadekar/PyTorch3D consumes the raw OBJ coordinates.
    # Keep the mesh vertices untouched and discard only that object transform.
    result.matrix_world = Matrix.Identity(4)
    return result


def emission_material(
    name: str,
    color: tuple[float, float, float, float],
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def camera_rotation(
    location: Vector,
    target: Vector,
    requested_up: Vector,
) -> Matrix:
    forward = (target - location).normalized()
    right = forward.cross(requested_up).normalized()
    up = right.cross(forward).normalized()
    return Matrix((right, up, -forward)).transposed().to_4x4()


def main() -> None:
    args = parse_args()
    if not args.obj.is_file():
        raise FileNotFoundError(args.obj)
    args.output.mkdir(parents=True, exist_ok=True)
    clear_scene()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.view_settings.look = "None"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("ProjectionWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world

    geometry = import_obj(args.obj)
    geometry.data.materials.clear()
    geometry.data.materials.append(
        emission_material("BlackSilhouette", (0.0, 0.0, 0.0, 1.0))
    )

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "SadekarTFSOrthographicCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = args.ortho_scale
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    scene.camera = camera

    distance = 5.0
    frames = [
        (Vector((0.0, 0.0, distance)), Vector((0.0, 1.0, 0.0))),
        (Vector((distance, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))),
        (Vector((0.0, distance, 0.0)), Vector((0.0, 0.0, 1.0))),
        (Vector((0.0, 0.0, -distance)), Vector((0.0, 1.0, 0.0))),
        (Vector((-distance, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))),
        (Vector((0.0, -distance, 0.0)), Vector((0.0, 0.0, 1.0))),
    ]
    for index, (location, up) in enumerate(frames):
        camera.location = location
        rotation = camera_rotation(location, Vector((0.0, 0.0, 0.0)), up)
        camera.matrix_world = rotation
        camera.location = location
        scene.render.filepath = str(args.output / f"blender_view_{index}.png")
        bpy.ops.render.render(write_still=True)

    blend_path = args.output / "sadekar_projection_audit.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(args.output)


if __name__ == "__main__":
    main()
