#!/usr/bin/env python3
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BENCH = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_globalconsistent_tfs_20260614/results/aaai2027_paper_20260615/e1_benchmark_final")
OUT = ROOT / "results/sadekar_style_main_comparison_20260705/blender_simulation"

SETS = [
    "setA_owl_lobster_ant",
    "setB_lobster_beetle_bat",
    "setC_turtle_scorpion_eagle",
    "setD_dragon_trex_scorpion",
]


def look_at(obj, target):
    loc = obj.location
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_mat(name, color, roughness=0.65):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_plane(name, loc, rot, scale, mat):
    bpy.ops.mesh.primitive_plane_add(size=2, location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    return obj


def import_obj(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    after = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not after:
        raise RuntimeError(f"No mesh imported from {path}")
    if len(after) == 1:
        return after[0]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in after:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = after[0]
    bpy.ops.object.join()
    return bpy.context.object


def normalize_object(obj, mat):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8
    dims = Vector((max(v.x for v in bbox) - min(v.x for v in bbox),
                   max(v.y for v in bbox) - min(v.y for v in bbox),
                   max(v.z for v in bbox) - min(v.z for v in bbox)))
    max_dim = max(dims.x, dims.y, dims.z)
    scale = 0.46 / max_dim if max_dim > 0 else 1.0
    obj.location -= center
    obj.scale = (-scale, -scale, scale)
    # Align Blender wall shadows with the paper view order:
    # left wall = view0, back wall = view1, floor = view2.
    obj.rotation_euler[2] = math.radians(90)
    obj.location = (0, 0, -0.18)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def setup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 650
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (1, 1, 1)

    wall_left = make_mat("left_wall", (0.86, 0.90, 0.56, 1))
    wall_back = make_mat("back_wall", (0.54, 0.42, 0.63, 1))
    floor_mat = make_mat("floor", (0.66, 0.70, 0.75, 1))
    model_mat = make_mat("model_red", (0.62, 0.16, 0.08, 1), 0.45)

    # Strict corner: left screen x=-1.05, back screen y=1.05, floor z=-0.70.
    add_plane("left_screen_x", (-1.05, 0, 0.15), (0, math.radians(90), 0), (0.95, 0.95, 1), wall_left)
    add_plane("back_screen_y", (0, 1.05, 0.15), (math.radians(90), 0, 0), (0.95, 0.95, 1), wall_back)
    add_plane("floor_z", (0, 0, -0.70), (0, 0, 0), (0.95, 0.95, 1), floor_mat)

    # Three near-parallel hard-shadow lights. This is closer to the paper-style
    # demonstration than soft area-light photography.
    lights = [
        ("to_left_wall", (4.0, 0.05, 0.12), (-1.05, 0, 0.15), 2.4, 0.01),
        ("to_back_wall", (0.05, -4.0, 0.12), (0, 1.05, 0.15), 2.4, 0.01),
        ("to_floor", (0.0, 0.0, 4.2), (0, 0, -0.70), 2.0, 0.01),
    ]
    for name, loc, target, power, size in lights:
        bpy.ops.object.light_add(type="SUN", location=loc)
        light = bpy.context.object
        light.name = name
        light.data.energy = power
        light.data.angle = size
        look_at(light, target)

    bpy.ops.object.camera_add(location=(2.65, -3.35, 1.75), rotation=(0, 0, 0))
    cam = bpy.context.object
    look_at(cam, (0, 0, 0))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.55
    scene.camera = cam
    return model_mat


def render_one(set_name):
    model_mat = setup_scene()
    obj = import_obj(BENCH / set_name / "ours_fixed/model/sample_0_export.obj")
    normalize_object(obj, model_mat)
    out = OUT / f"{set_name}_blender_corner_sim.png"
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f"{set_name}_blender_scene.blend"))
    bpy.ops.render.render(write_still=True)
    print(out)


def main():
    for set_name in SETS:
        render_one(set_name)


if __name__ == "__main__":
    main()
