#!/usr/bin/env python3
"""Render a physical two-wall verification for the quality-first STL exports."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
OUT = SRC / "blender_physical_corner_verification"
SCRIPT = OUT / "render_two_wall.py"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for manifest in sorted(SRC.glob("*/CASE_MANIFEST.json")):
        case = manifest.parent.name
        stl = next(manifest.parent.glob("*_upright_saving_10cm_256.stl"))
        items.append((case, str(stl)))
    payload = repr(items)
    SCRIPT.write_text(f'''\
import bpy
from pathlib import Path
from mathutils import Vector

OUT = Path(r"{OUT}")
ITEMS = {payload}

def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

def material(name, color, roughness=0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Roughness"].default_value = roughness
    return mat

def cube(name, location, dimensions, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj

def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()

def setup(case, path):
    clear()
    wall_l = material("left_wall", (0.88, 0.88, 0.72, 1))
    wall_r = material("right_wall", (0.68, 0.57, 0.74, 1))
    floor = material("floor", (0.34, 0.37, 0.40, 1))
    sculpture = material("sculpture", (0.08, 0.42, 0.22, 1), 0.62)
    # The two planes are exactly perpendicular and lie behind the projected
    # silhouette receivers.  The model remains at the calibrated origin.
    cube("left_receiver_x", (62, 0, 0), (1.0, 180, 160), wall_l)
    cube("right_receiver_y", (0, 62, 0), (180, 1.0, 160), wall_r)
    cube("floor", (0, 0, -46), (180, 180, 1.0), floor)
    bpy.ops.wm.stl_import(filepath=path)
    obj = bpy.context.object
    obj.name = case
    obj.data.materials.append(sculpture)
    bpy.ops.object.camera_add(location=(-175, -175, 105))
    camera = bpy.context.object
    look_at(camera, (17, 17, -1))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 205
    bpy.context.scene.camera = camera
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 760
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.035, 0.035)
    scene.view_settings.look = "AgX - Medium High Contrast"

def add_parallel_light(axis):
    # Sun lights are parallel by construction, matching the orthographic
    # projection model used for candidate evaluation.
    if axis == "x":
        pos, target = (-1, 0, 0), (62, 0, 0)
    else:
        pos, target = (0, -1, 0), (0, 62, 0)
    bpy.ops.object.light_add(type="SUN", location=pos)
    light = bpy.context.object
    light.data.energy = 3.0
    light.data.angle = 0.003
    look_at(light, target)

def render(case, path, mode):
    setup(case, path)
    if mode in ("left", "both"):
        add_parallel_light("x")
    if mode in ("right", "both"):
        add_parallel_light("y")
    bpy.context.scene.render.filepath = str(OUT / f"{{case}}_{{mode}}.png")
    bpy.ops.render.render(write_still=True)

for case, path in ITEMS:
    render(case, path, "left")
    render(case, path, "right")
    render(case, path, "both")
''', encoding="utf-8")
    subprocess.run(["blender", "-b", "--python", str(SCRIPT)], check=True)


if __name__ == "__main__":
    main()
