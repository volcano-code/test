#!/usr/bin/env python3
"""Render one STL for diagnostic inspection with Blender.

Usage:
  blender -b --python experiments/render_inspection_stl.py -- input.stl output.png
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def arguments() -> tuple[Path, Path]:
    separator = sys.argv.index("--")
    values = sys.argv[separator + 1 :]
    if len(values) != 2:
        raise SystemExit("Expected: input.stl output.png")
    return Path(values[0]).resolve(), Path(values[1]).resolve()


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    source, output = arguments()
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.stl_import(filepath=str(source))
    mesh = bpy.context.object

    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    mesh.location = (0, 0, 0)
    corners = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    low = Vector(tuple(min(point[i] for point in corners) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in corners) for i in range(3)))
    mesh.scale *= 2.6 / max(high - low)
    mesh.rotation_euler = (
        math.radians(8),
        math.radians(-6),
        math.radians(-24),
    )

    material = bpy.data.materials.new("inspection")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.10, 0.58, 0.52, 1.0)
    shader.inputs["Roughness"].default_value = 0.62
    mesh.data.materials.append(material)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.world.color = (1.0, 1.0, 1.0)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 3.5
    camera.location = (4.8, -5.4, 3.7)
    point_at(camera, Vector((0, 0, 0)))

    for name, location, energy, size in (
        ("key", (4.5, -4.0, 6.0), 900, 4.5),
        ("fill", (-4.0, -2.0, 3.5), 450, 5.0),
        ("rim", (-2.0, 4.0, 5.0), 650, 4.0),
    ):
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.energy = energy
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = location
        point_at(light, Vector((0, 0, 0)))

    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


main()
