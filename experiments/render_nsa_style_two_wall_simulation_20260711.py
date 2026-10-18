"""Render verified two-wall ShadowForge STLs in an NSA-style corner scene.

This is a Blender simulation asset, not fabrication evidence.  Wall shadows
are physically cast by the imported STL under two spot lights; no shadow image
is composited onto the walls.
"""
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
OUT = SRC / "blender_nsa_style_simulation"
ITEMS = [
    "dark_knight_left__magic_scholar_right",
    "dark_mage_left__samurai_right",
    "fox_miko_left__violinist_right",
    "future_mecha_left__skater_right",
    "umbrella_miko_left__archer_right",
]


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.materials, bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def mat(name, color, roughness=0.65, metallic=0.0):
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return value


def cube(name, location, dimensions, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_spot(name, location, target):
    bpy.ops.object.light_add(type="SPOT", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = 400000
    light.data.color = (1.0, 0.96, 0.88)
    light.data.spot_size = 0.56
    light.data.spot_blend = 0.62
    light.data.shadow_soft_size = 0.18
    light.data.use_shadow = False
    point_at(light, target)
    return light


def add_sun(name, direction_target):
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    light = bpy.context.object
    light.name = name
    light.data.energy = 3.0
    light.data.angle = 0.020
    light.data.color = (1.0, 0.95, 0.86)
    light.data.angle = 0.035
    point_at(light, direction_target)
    return light


def link_receivers(light, name, objects):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        collection.objects.link(obj)
    light.light_linking.receiver_collection = collection


def setup_scene(case, stl):
    clear()
    wall = mat("warm gray plaster", (0.76, 0.75, 0.70, 1.0), 0.92)
    floor = mat("matte charcoal tabletop", (0.002, 0.0025, 0.003, 1.0), 0.62)
    stand = mat("low presentation plinth", (0.11, 0.12, 0.13, 1.0), 0.40, 0.05)
    print_material = mat("matte warm gray PLA", (0.31, 0.29, 0.26, 1.0), 0.67)

    # Strict 90-degree physical wall corner.  The STL remains in its verified
    # fixed coordinate system; walls lie in the calibrated +X/+Y directions.
    left_wall = cube("left_wall_x", (126.0, 0.0, 38.0), (1.2, 280.0, 190.0), wall)
    right_wall = cube("right_wall_y", (0.0, 126.0, 38.0), (280.0, 1.2, 190.0), wall)
    cube("table", (0.0, 0.0, -51.5), (280.0, 280.0, 2.0), floor)
    cube("display_plinth", (0.0, 0.0, -49.4), (72.0, 72.0, 2.2), stand)

    bpy.ops.wm.stl_import(filepath=str(stl))
    obj = bpy.context.object
    obj.name = f"{case}_printed_model"
    obj.data.materials.append(print_material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True

    # The physical spot lights create the wall shadows.  They are far enough
    # from a 10 cm object to keep perspective distortion modest while retaining
    # the bright circular pools visible in fabrication photographs.
    left_spot = add_spot("left_projection_light", (-130.0, 0.0, 8.0), (126.0, 0.0, 15.0))
    right_spot = add_spot("right_projection_light", (0.0, -130.0, 8.0), (0.0, 126.0, 15.0))
    link_receivers(left_spot, "left_projection_receivers", [left_wall, obj])
    link_receivers(right_spot, "right_projection_receivers", [right_wall, obj])
    # Parallel components retain the calibrated two-wall projection directions.
    left_sun = add_sun("left_parallel_shadow_light", (1.0, 0.0, 0.0))
    right_sun = add_sun("right_parallel_shadow_light", (0.0, 1.0, 0.0))
    link_receivers(left_sun, "left_parallel_receivers", [left_wall, obj])
    link_receivers(right_sun, "right_parallel_receivers", [right_wall, obj])

    bpy.ops.object.light_add(type="AREA", location=(-40.0, -55.0, 135.0))
    fill = bpy.context.object
    fill.name = "soft_camera_fill"
    fill.data.energy = 60
    fill.data.shape = "DISK"
    fill.data.size = 120.0
    point_at(fill, (0.0, 0.0, -12.0))

    bpy.ops.object.camera_add(location=(-310.0, -310.0, 155.0))
    camera = bpy.context.object
    point_at(camera, (32.0, 32.0, -10.0))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 338.0
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.get_devices()
        for device in preferences.devices:
            device.use = True
    except Exception:
        # Rendering remains valid on CPU when a local CUDA/OptiX backend is
        # unavailable; only the runtime changes.
        pass
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.035, 0.040)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 1.1
    scene.render.image_settings.color_mode = "RGB"


OUT.mkdir(parents=True, exist_ok=True)
for case in ITEMS:
    stl = SRC / case / f"{case}_upright_saving_10cm_256.stl"
    if not stl.exists():
        continue
    setup_scene(case, stl)
    bpy.context.scene.render.filepath = str(OUT / f"{case}_nsa_style_simulated.png")
    bpy.ops.render.render(write_still=True)
